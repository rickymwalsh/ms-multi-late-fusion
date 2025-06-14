
import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from skimage import morphology
import skimage.filters
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging

from im_utils import sitk_to_numpy, new_image_from_ref, resample_to_ref, get_bbox_bounds


# TODO: Restrict to relevant preds - will only process the mean CCs, no need for other CCs
# TODO: decide on thresholds - different threshold for different subgroup (but always mean?)


def load_reorient(im_path, orientation='LAS'):
    im = sitk.ReadImage(im_path)
    return sitk.DICOMOrient(im, orientation)


def load_ims(pred_dir, pred_names, orientation='LAS'):
    """ Load the prediction images that exist, re-orient to a given orientation and return as a dictionary with their
    filenames as keys."""
    pred_paths = {pred_name: pred_dir / pred_name for pred_name in pred_names if (pred_dir / pred_name).exists()}
    preds = {name: load_reorient(p, orientation) for name, p in pred_paths.items()}
    return preds


def get_iou(gt, pred):
    # Compute IoU: intersection / union
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union != 0 else 0


def get_ccs_and_save(im, thresh, output_path):
    arr = sitk_to_numpy(im)

    # Binarise and get connected components
    arr_bin = arr > thresh
    cc = label(arr_bin)
    cc_im = new_image_from_ref(cc, im)
    sitk.WriteImage(cc_im, output_path)


def generate_ccs(data_dir: Path, thresholds: List, pred_names: List, cc_names: List):
    """ Generate connected components for one case, for each prediction image and at each threshold.
    Args:
        data_dir (Path): Path to the directory containing the predictions.
        thresholds (list): List of thresholds to use for generating connected components.
        pred_names (list): List of prediction image names.
        cc_names (list): List of connected component filenames to save.
    Returns: None
        Saves the CCs in the following directory structure: output_dir/ccs/thr-{thresh}/{cc_name}
        """
    out_dir = data_dir / 'ccs'
    out_dir.mkdir(exist_ok=True, parents=True)

    preds = load_ims(data_dir, pred_names, 'LAS')

    for thresh in thresholds:
        outdir_thr = out_dir / f'thr-{thresh}'
        outdir_thr.mkdir(exist_ok=True)

        for cc_name in cc_names:
            pred = extract_from_dict(preds, [cc_name.replace('_cc.nii.gz', '')])[0]
            if pred and not (outdir_thr / cc_name).exists():
                get_ccs_and_save(pred, thresh, outdir_thr / cc_name)


def clean_up_ccs(input_dir: Path, cc_names: List, thresholds: List, reference_threshold: float):
    """ For the lowest thresholds (10^-6), we get thousands of small connected components. While these could be
    removed by the posthoc model because of their low probability and small volume, they significantly slow down the
    feature extraction process. For this reason, we remove them here, by only keeping connected components that also
    exist at a higher reference threshold, so the lower threshold just gives a larger segmentation of existing preds.
    Args:
        input_dir (Path): Path to the directory containing the connected components.
        cc_names (list): List of saved connected component filenames.
        thresholds (list): List of thresholds for the connected components we want to clean up.
        reference_threshold (str): The threshold for the reference connected components.
    """
    # Load the reference CCs
    ref_ccs = load_ims(input_dir / 'ccs' / f'thr-{reference_threshold}', cc_names, 'LAS')
    # Load the CCs for each threshold
    for thresh in thresholds:
        thresh_dir = input_dir / 'ccs' / f'thr-{thresh}'
        if not thresh_dir.is_dir():
            continue
        old_ccs = load_ims(thresh_dir, cc_names, 'LAS')
        for cc_name, old_cc_im in old_ccs.items():
            if old_cc_im is None:
                continue
            new_filename = cc_name.replace('_cc.nii.gz', '_old_cc.nii.gz')
            if (thresh_dir / new_filename).exists():
                continue
            old_cc_arr = sitk_to_numpy(old_cc_im)  # Get the current connected component array
            ref_cc_arr = (sitk_to_numpy(ref_ccs[cc_name]) > 0).astype(np.uint8)  # reference connected component array
            existing_ccs = np.unique(old_cc_arr * ref_cc_arr)  # Get the CCs that exist in both arrays
            new_cc_arr = old_cc_arr * np.isin(old_cc_arr, existing_ccs).astype(np.uint8)
            # Rename the old file
            (thresh_dir / cc_name).rename(thresh_dir / f'{cc_name.replace("_cc.nii.gz", "_old_cc.nii.gz")}')
            if (thresh_dir / cc_name).exists() or \
                    not (thresh_dir / f'{cc_name.replace("_cc.nii.gz", "_old_cc.nii.gz")}').exists():
                raise ValueError(f"Error renaming file {thresh_dir / cc_name} to "
                                 f"{thresh_dir / cc_name.replace('_cc.nii.gz', '_old_cc.nii.gz')}.")
            else:
                # Save the new connected component image
                new_cc_im = new_image_from_ref(new_cc_arr, old_cc_im)
                sitk.WriteImage(new_cc_im, thresh_dir / cc_name)


def generate_neighbourhoods(input_dir: Path, cc_names: List):
    """ Generate neighbourhoods for one case, for each prediction connected component image.
    This may not be perfect, because the neighbourhoods for two lesions may overlap, and here we would only take the
    value of the higher CC ID. For the purpose of feature engineering however, and in the interest of saving time,
    this is a better approach (> 90% time reduction vs. looping through each CC ID and dilating).
    Args:
        input_dir (Path): Path to the directory containing the predictions.
        cc_names (list): List of saved connected component filenames.
    Returns: None
        Saves the CCs in the following directory structure: output_dir/subj_id/thr-{thresh}/{out_name}
        """
    cc_dir = input_dir / 'ccs'
    out_dir = input_dir / 'nhoods'
    for thresh_dir in cc_dir.iterdir():
        if not thresh_dir.is_dir() or not thresh_dir.name.startswith('thr-'):
            continue
        thresh = thresh_dir.name.replace('thr-', '')
        out_dir_thr = out_dir/ f'thr-{thresh}'
        out_dir_thr.mkdir(exist_ok=True, parents=True)

        ccs = load_ims(thresh_dir, cc_names, 'LAS')
        # Loop over the CCs (T2 CCs, other seq, mean CCs)
        for cc_filename, cc_im in ccs.items():
            if cc_im is None:
                continue
            out_fpath = out_dir_thr / cc_filename.replace('_cc.nii.gz', '_nhood.nii.gz')
            if out_fpath.exists():
                continue
            cc_arr = sitk_to_numpy(cc_im)  # Get the connected component array
            # Get the neighbourhood of the connected component
            neighbourhood = skimage.filters.rank.maximum(cc_arr, morphology.ball(5))
            # Save to file
            neighbourhood_im = new_image_from_ref(neighbourhood, cc_im)
            sitk.WriteImage(neighbourhood_im, out_fpath)


def summarise_probabilities(im_arr, cc):
    rel_probs = im_arr[cc > 0]
    if len(rel_probs) == 0 or rel_probs.max() == 0:
        return {
            'max': 0.0,
            'min': 0.0,
            'mean': 0.0,
            'median': 0.0,
            '10th': 0.0,
            '20th': 0.0,
            '90th': 0.0,
            '95th': 0.0,
            '99th': 0.0
        }
    else:
        return {
            'max': np.max(rel_probs),
            'min': np.min(rel_probs),
            'mean': np.mean(rel_probs),
            'median': np.median(rel_probs),
            '10th': np.percentile(rel_probs, 10),
            '20th': np.percentile(rel_probs, 20),
            '90th': np.percentile(rel_probs, 90),
            '95th': np.percentile(rel_probs, 95),
            '99th': np.percentile(rel_probs, 99)
        }


def lesion_load(im_arr, spacing):
    return np.sum(im_arr > 0) * np.prod(spacing)


def extract_from_dict(d, key_patterns):
    def extract_single(d_, key_pattern):
        matches = [v for k, v in d_.items() if key_pattern in k]
        if len(matches) == 0:
            return None
        elif len(matches) > 1:
            raise ValueError(f"Multiple matches found for {key_pattern}: {[k for k in d_.keys() if key_pattern in k]}")
        return matches[0]

    return [extract_single(d, key_pattern) for key_pattern in key_patterns]


def analyse_adjacent_slices(pred_arr: np.ndarray, cc_arr: np.ndarray, slice_num: int, spacing: tuple):
    """ Extract statistics for the adjacent slices given a current lesion and current slice.
    Args:
        pred_arr (np.ndarray): The prediction array for the current sequence (can be either full, or masked to the
                                current connected component of interest)
        cc_arr (np.ndarray): The current connected component array (i.e. a single 3D connected component)
        slice_num (int): The slice number of the current slice, the two adjacent slices will be slice_num - 1 and \
                         slice_num + 1
        spacing (tuple): The spacing of the image, used to calculate lesion volumes

    Returns:
        lesion_vol_slice1, max_prob_slice1, mean_prob_slice1, lesion_vol_slice2, max_prob_slice2, mean_prob_slice2
        Slice 1 is the slice containing the higher volume of the current connected component
    """
    # Get the slice numbers of the adjacent slices
    slice_num1, slice_num2 = slice_num - 1, slice_num + 1
    # Handle the case where the current slice is at the edge of the image
    if slice_num1 < 0:
        slice_num1 = slice_num
    if slice_num2 >= pred_arr.shape[0]:
        slice_num2 = slice_num
    # Get the slices from the prediction array
    slice1, slice2 = pred_arr[slice_num1], pred_arr[slice_num2]
    # Get the connected components from the connected component array
    cc1, cc2 = cc_arr[slice_num1], cc_arr[slice_num2]
    # Get the lesion volumes
    vol1, vol2 = lesion_load(cc1, spacing), lesion_load(cc2, spacing)
    # Get the probabilities
    probs1, probs2 = summarise_probabilities(slice1, cc1), summarise_probabilities(slice2, cc2)

    # Handle the case where the current slice is at the edge of the image
    if slice_num1 == slice_num:
        return {'vol_slice1': vol2, 'max_prob_slice1': probs2['max'], 'mean_prob_slice1': probs2['mean'],
                'vol_slice2': None, 'max_prob_slice2': None, 'mean_prob_slice2': None}
    if slice_num2 == slice_num:
        return {'vol_slice1': vol1, 'max_prob_slice1': probs1['max'], 'mean_prob_slice1': probs1['mean'],
                'vol_slice2': None, 'max_prob_slice2': None, 'mean_prob_slice2': None}

    # Return the features for the slice with the higher lesion volume first
    if vol1 >= vol2:
        return {'vol_slice1': vol1, 'max_prob_slice1': probs1['max'], 'mean_prob_slice1': probs1['mean'],
                'vol_slice2': vol2, 'max_prob_slice2': probs2['max'], 'mean_prob_slice2': probs2['mean']}
    else:
        return {'vol_slice1': vol2, 'max_prob_slice1': probs2['max'], 'mean_prob_slice1': probs2['mean'],
                'vol_slice2': vol1, 'max_prob_slice2': probs1['max'], 'mean_prob_slice2': probs1['mean']}


def two_seq_string(seq1, seq2):
    """ Return a string representing the two sequences in a consistent order.
    T2 > STIR > MP2RAGE > PSIR > mean """
    seqs = ['T2', 'STIR', 'MP2RAGE', 'PSIR', 'mean']
    if seqs.index(seq1) > seqs.index(seq2):
        return f'{seq2}_{seq1}'
    return f'{seq1}_{seq2}'


def get_local_contrast(im_arr: np.ndarray, mask: np.ndarray, sc_seg_arr: np.ndarray,
                       neighbourhood_arr: Optional[np.ndarray]) -> float:
    """ Calculate the contrast of the lesion vs. the surrounding neighbourhood (dilated and masked by SC)
    Args:
        im_arr (np.ndarray): The image array with intensities
        mask (np.ndarray): The binary mask of the lesion
        sc_seg_arr (np.ndarray): The SC segmentation array
        neighbourhood_arr (np.ndarray): The neighbourhood array, if it has been pre-computed, to calculate contrast.
    Returns:
        float: The contrast value: (mean_lesion_intensity - mean_neighbourhood_intensity) / mean_neighbourhood_intensity
    """
    # Determine the structuring element to use, based on whether the array is 2D or 3D
    if mask.ndim not in [2, 3]:
        raise ValueError("[get_local_contrast] mask must be either 2D or 3D")
    if neighbourhood_arr is None:
        footprint = morphology.disk(5) if mask.ndim == 2 else morphology.ball(5)
        # Dilate the mask
        mask_neighbourhood = morphology.binary_dilation(mask, footprint)
    else:
        mask_neighbourhood = (neighbourhood_arr > 0).astype(np.uint8)
    # Mask the dilated mask to the SC region
    mask_neighbourhood = mask_neighbourhood * (sc_seg_arr > 0)
    # Remove the initial area of the lesion
    mask_neighbourhood = mask_neighbourhood * (mask == 0)
    if mask_neighbourhood.sum() == 0:
        return np.nan

    # Calculate the mean intensity of the lesion and the neighbourhood
    mean_lesion = im_arr[mask > 0].mean()
    mean_neighbourhood = im_arr[mask_neighbourhood > 0].mean()
    if mean_neighbourhood == 0:
        return np.nan

    return (mean_lesion - mean_neighbourhood) / mean_neighbourhood


def get_intensity_stats(arr: np.ndarray, cc: np.ndarray, sc_seg_arr: np.ndarray,
                        neighbourhood_arr: Optional[np.ndarray] = None, slice_idx: Optional[int] = None,
                        fov_arr: Optional[np.ndarray] = None) -> Dict:
    """ Calculate intensity statistics for the connected component in the image.
    Args:
        arr (np.ndarray): The 3D image array. Should be the same size as the connected component array, except an extra
                          axis at position 0.
        cc (np.ndarray): A 3D array with a single mask or connected component.
        sc_seg_arr (np.ndarray): The 3D SC segmentation array, used to mask the intensity statistics to the SC region.
        neighbourhood_arr (np.ndarray): The 3D neighbourhood array, if it has been pre-computed, to calculate contrast.
        slice_idx (int): The slice index to use to extract a slice from the 3D image array (and other arrays).
                         If None, the entire 3D array is used.
        fov_arr (np.ndarray): The 3D field of view array, used to check if the CC is within the field of view of the
                            anat image, and if not then return None for all statistics.
    Returns:
        dict containing the following statistics: mean_intensity, min_intensity, max_intensity, median_intensity,
        5th_perc_intensity, 95th_perc_intensity, contrast_vs_sc, mean_vs_sc_std, contrast_vs_neighbourhood
    """
    if fov_arr is not None:
        in_fov_pct = fov_arr[cc > 0].sum() / (cc > 0).sum()
    else:
        in_fov_pct = 1.0

    if any([arr is None, cc is None, sc_seg_arr is None, sc_seg_arr.sum() == 0, in_fov_pct < 0.5]):
        return {'mean_intensity': None, 'min_intensity': None, 'max_intensity': None, 'median_intensity': None,
                '5th_perc_intensity': None, '95th_perc_intensity': None, 'contrast_vs_sc': None,
                'mean_vs_sc_std': None, 'contrast_vs_neighbourhood': None}
    # Get global mean & std deviation of the SC intensities
    mean_sc, std_sc = arr[sc_seg_arr > 0].mean(), arr[sc_seg_arr > 0].std()

    if slice_idx is not None:
        # Get the slice of the image and connected component
        arr = arr[slice_idx]
        sc_seg_arr = sc_seg_arr[slice_idx]
        neighbourhood_arr = neighbourhood_arr[slice_idx] if neighbourhood_arr is not None else None

    mask = (cc > 0).astype(np.uint8)
    # Get intensities in the lesion and in the SC
    masked_arr = arr[mask > 0]
    masked_sc = arr[sc_seg_arr > 0]
    # Calculate the contrast statistics
    if len(masked_sc) > 0 and masked_sc.max() > 0:
        contrast_vs_sc = (masked_arr.mean() - masked_sc.mean()) / masked_sc.mean()
        mean_vs_sc_std = masked_arr.mean() / masked_sc.std()
    else:
        contrast_vs_sc = np.nan
        mean_vs_sc_std = np.nan
    contrast_vs_neighbourhood = get_local_contrast(arr, mask, sc_seg_arr, neighbourhood_arr)

    # Normalise the intensities by the SC mean and std deviation
    masked_arr = (masked_arr - mean_sc) / std_sc

    return {
        'mean_intensity': masked_arr.mean(),
        'min_intensity': masked_arr.min(),
        'max_intensity': masked_arr.max(),
        'median_intensity': np.median(masked_arr),
        '5th_perc_intensity': np.percentile(masked_arr, 5),
        '95th_perc_intensity': np.percentile(masked_arr, 95),
        'contrast_vs_sc': contrast_vs_sc,
        'mean_vs_sc_std': mean_vs_sc_std,
        'contrast_vs_neighbourhood': contrast_vs_neighbourhood
    }


class FeatureProcessor:
    def __init__(self, process_slicewise: bool, pred_dir_parent: Path, anat_dir: Path, thresholds: List, pred_filenames: List,
                 cc_filenames: List, fov_filenames: List[str], metadata_df: Optional[pd.DataFrame] = None,
                 metadata_cols: Optional[List[str]] = None,
                 ):
        self.process_slicewise = process_slicewise
        self.pred_dir = pred_dir_parent / 'tmp'
        self.anat_dir = anat_dir
        self.thresholds = thresholds
        self.pred_filenames = pred_filenames
        self.cc_filenames = cc_filenames
        self.fov_filenames = fov_filenames
        self.metadata_df = metadata_df
        self.metadata_cols = metadata_cols

        if self.metadata_df is not None and self.metadata_cols:
            # Check there is a unique row per 'subject'
            if not self.metadata_df['subject'].is_unique:
                raise ValueError("Metadata dataframe must have a unique row per subject.")

        self.current_sub = anat_dir.name
        self.metadata_sub = self.extract_metadata(self.current_sub)
        self.nhood_dir = self.pred_dir / 'nhoods'
        self.cc_dir = self.pred_dir / 'ccs'
        self.out_file = self.pred_dir / 'features.csv'

        self.seqnames = ['T2', 'STIR', 'MP2RAGE', 'PSIR', 'mean']
        self.anat_seqnames = ['T2', 'STIR', 'MP2RAGE', 'PSIR']

        self.cc_arrs_names = None
        self.features_df = None
        self.anat_im = None  # target image for resampling if necessary
        self.spacing = None  # target spacing
        self.mid_slice_num = None
        self.upper_sc_limit = None

    def extract_metadata(self, sub_id: str) -> Dict:
        """ Extract metadata for a given subject ID. """
        if self.metadata_cols is None:
            return {}
        sub_row = self.metadata_df[self.metadata_df['subject'] == sub_id]
        if not sub_row.empty:
            return {col: sub_row[col].values[0] for col in self.metadata_cols}
        else:
            logging.warning(f"No metadata found for subject {sub_id}")
            return {}

    def update_anat_im(self):
        """ Load the target anat image, to which all predictions should be resampled. """
        fpath = list(self.anat_dir.glob('*T2.nii.gz'))[0]
        self.anat_im = load_reorient(fpath)
        self.spacing = self.anat_im.GetSpacing()

    def load_and_extract_ims(self, dirpath: Path, filenames: List[str], filename_patterns: List[str],
                             interpolator=sitk.sitkLinear) -> List[Tuple[np.ndarray, str]]:
        """ Load images for a given case and extract the relevant images based on the filename patterns.
        Args:
            dirpath (Path): Path to the subject directory containing the images.
            filenames (List[str]): List of filenames to load from the subject directory. e.g. ['T2_pred.nii.gz',...]
            filename_patterns (List[str]): List of filename patterns to extract. e.g. ['T2', 'STIR', 'mean', ...]
            interpolator (SimpleITK Interpolator): Interpolator to use for resampling the images to the ref image.
        Returns:
            List of tuples, with each tuple being (image_arr, filename_pattern) only for images which exist.
        """
        ims = load_ims(dirpath, filenames)
        extracted_ims = extract_from_dict(ims, filename_patterns)
        # Re-sample all images to the GT image to ensure same size, spacing, etc.
        extracted_ims = [resample_to_ref(im, self.anat_im, interpolator=interpolator) if im is not None else None
                         for im in extracted_ims]
        extracted_ims = [sitk_to_numpy(im) if im is not None else None for im in extracted_ims]
        # Filter out None values and return the images and their corresponding filename patterns
        return [(im, name) for im, name in zip(extracted_ims, filename_patterns) if im is not None]

    def update_sc_info(self, sc_seg_arr):
        """ Update the spinal cord position information for the current subject."""
        # Get the bounding box of the spinal cord segmentation
        sc_bbox = get_bbox_bounds(sc_seg_arr)
        self.mid_slice_num = (sc_bbox[0][0] + sc_bbox[0][1]) / 2
        self.upper_sc_limit = sc_bbox[2][1]  # Images in LAS+ orientation, so last axis [2] for I-S and [1] for Superior

    def loop_ccs(self) -> Tuple[np.ndarray, str, List[int]]:
        # Keep these in memory to avoid having to load several times during processing
        if self.cc_arrs_names is None:
            self.cc_arrs_names = {
                thr: self.load_and_extract_ims(self.cc_dir / f'thr-{thr}', self.cc_filenames,
                                               self.seqnames, interpolator=sitk.sitkNearestNeighbor)
                for thr in self.thresholds
            }
        # self.cc_arrs_names - all CC arrays for the given subject, along with the corresponding seq name
        #  e.g., {0.0001: [(t2_arr, 'T2'), (stir_arr, 'STIR') or (None, 'STIR'), ...], 0.001: [...], ...}

        for thr in self.thresholds:
            for cc_arr, cc_name in self.cc_arrs_names[thr]:
                cc_ids = np.unique(cc_arr)
                if len(cc_ids) == 1 and cc_ids[0] == 0:
                    continue
                yield cc_arr, cc_name, cc_ids[1:], thr

    def load_nhood_arr(self, cc_name_pattern: str, thresh: float) -> Optional[np.ndarray]:
        """ Load the neighbourhood array for a given case and sequence name. """
        cc_filename = [f for f in self.cc_filenames if cc_name_pattern in f][0]
        nhood_path = self.nhood_dir / f'thr-{thresh}' / f'{cc_filename.replace("_cc.nii.gz", "_nhood.nii.gz")}'
        if nhood_path.exists():
            nhood_im = load_reorient(nhood_path, 'LAS')
            nhood_im = resample_to_ref(nhood_im, self.anat_im, interpolator=sitk.sitkNearestNeighbor)
            nhood_arr = sitk_to_numpy(nhood_im)
            return nhood_arr
        else:
            return None

    def update_row(self, cc: int, seqname: str, threshold: float, slice_idx: Optional[int] = None,
                   seq_features: Optional[Dict[str, Dict]] = None,
                   general_features: Optional[Dict] = None) -> Dict:
        """ Structure several features and feature types into a single row (dict).
        Args:
            cc (int): The connected component ID
            seqname (str): The sequence name from which we took the current lesion
            threshold (float): The threshold used to generate the connected component
            slice_idx (int): The slice index, if processing slice-wise
            seq_features (List[Dict[str, Dict]]): Features specific to a sequence, where the sequence name should be
                        prepended  e.g., {'T2': {'feat1': value1, 'feat2': value2}}, {'STIR': {...}}
            general_features (Dict): General features that are not specific to a sequence. e.g.
                                    e.g. {'dist_to_mid': 5.0, ...}
        Returns:
            A dictionary containing all features for the current connected component.
            e.g. {'case_id': 'sub-001', 'cc_id': 1, 'slice_num': 2, 'dist_to_mid': 5.0,
                  'T2_feat1': value1, 'T2_feat2': value2, 'STIR_feat1': value3, ...}
            """
        seq_features_collapsed = {}
        if seq_features:
            for seq_name, features in seq_features.items():
                for k, v in features.items():
                    seq_features_collapsed[f'{seq_name}_{k}'] = v
        general_features = {} if general_features is None else general_features
        row = {
            'case_id': self.current_sub,
            'cc_id': cc,
            'seq': seqname,
            'threshold': threshold,
            **seq_features_collapsed,
            **general_features
        }
        if slice_idx is not None:
            row['slice_num'] = slice_idx
        return row

    def gather_intensity_stats(self, sc_seg_arr: np.ndarray) -> pd.DataFrame:
        """ Gather intensity statistics for the current case. """
        # Load the anat images
        anat_filenames = [self.anat_dir.glob(f'*{seq}.nii.gz') for seq in self.anat_seqnames]
        anat_filenames = [f.name for sublist in anat_filenames for f in sublist]
        anat_arrs_names = self.load_and_extract_ims(self.anat_dir, anat_filenames, self.anat_seqnames)
        fov_names = [name + '_FOV' for name in self.anat_seqnames]
        fov_arrs_names = self.load_and_extract_ims(self.pred_dir, self.fov_filenames, fov_names,
                                                   interpolator=sitk.sitkNearestNeighbor)
        data = []
        for cc_arr, cc_name, cc_ids, thresh in self.loop_ccs():
            nhood_arr = self.load_nhood_arr(cc_name, thresh)
            for anat_arr, anat_name in anat_arrs_names:
                # Extract the FOV which matches the anat_name
                fov_arr = [fov for fov, name in fov_arrs_names if name.replace('_FOV', '') == anat_name]
                fov_arr = fov_arr[0] if fov_arr else None
                for cc_id in cc_ids:
                    cc_bin = (cc_arr == cc_id).astype(np.uint8)
                    if not self.process_slicewise:
                        # Get the intensity statistics for the entire connected component
                        nhood_arr_bin = (nhood_arr == cc_id).astype(np.uint8) if nhood_arr is not None else None
                        intensity_stats = get_intensity_stats(anat_arr, cc_bin, sc_seg_arr, nhood_arr_bin,
                                                              slice_idx=None, fov_arr=fov_arr)

                        row = self.update_row(cc_id, cc_name, thresh, seq_features={anat_name: intensity_stats})
                        data.append(row)
                    else:
                        # Get the intensity statistics for each slice
                        for i_x, cc_slice in enumerate(cc_bin):
                            if np.sum(cc_slice) == 0:
                                continue
                            intensity_stats = get_intensity_stats(anat_arr, cc_slice, sc_seg_arr, slice_idx=i_x,
                                                                  fov_arr=fov_arr)
                            row = self.update_row(cc_id, cc_name, thresh, slice_idx=i_x,
                                                  seq_features={anat_name: intensity_stats})
                            data.append(row)
        # There will be duplicate rows per sub_id, cc_id, seqname, thresh and slice_idx, one row for each anat image
        #   - where this happens, we want to keep both sets of columns for that row
        df = pd.DataFrame(data)
        # take the first non-null value for each column
        id_cols = ['case_id', 'cc_id', 'seq', 'threshold']
        if self.process_slicewise:
            id_cols.append('slice_num')
        combined_df = df.groupby(id_cols).agg('first').reset_index()
        return combined_df

    def get_position_stats(self, cc, slice_idx=None):
        """ Get the position statistics for the connected component.
        Args:
            cc (np.ndarray): The connected component array.
            slice_idx (int): The slice index to use to extract a slice from the 3D image array (and other arrays).
                             If None, the entire 3D array is used.
        Returns:
            dict containing the following statistics:
                dist_to_mid - distance (in mm) from current slice (if 2D) or mid lesion slice (if 3D) to the
                                middle slice of the spinal cord,
                dist_to_top - distance (in mm) from the uppermost voxel of the connected component to the
                                top of the spinal cord segmentation
        """
        uppermost_lesion_voxel = np.where(cc > 0)[-1].max()  # [-1] to extract i-S axis
        # Get the distance to the top of the spinal cord segmentation in the image
        dist_to_top = (self.upper_sc_limit - uppermost_lesion_voxel) * self.spacing[2]
        # Get the distance to the middle slice
        if self.process_slicewise:
            dist_to_mid = abs(slice_idx - self.mid_slice_num) * self.spacing[0]
        else:
            # Get middle slice of lesion
            cc_bbox = get_bbox_bounds(cc)
            mid_lesion_slice = (cc_bbox[0][0] + cc_bbox[0][1]) / 2  # Allow for non-integer values
            dist_to_mid = abs(mid_lesion_slice - self.mid_slice_num) * self.spacing[0]

        return {
            'dist_to_mid': dist_to_mid,
            'dist_to_top': dist_to_top
        }

    def process_current_2d_or_3d(self, pmaps_arr, cc_bin, cc_arr, seq_name, slice_idx=None):
        """ Process the current connected component and extract features for the current slice or the 3D component.
        Args:
            pmaps_arr - The prediction array for the current sequence
            cc_bin - The binary mask of the current connected component for the current slice or 3D component
            cc_arr - The full 3D connected component array for the current sequence
            seq_name - The name of the current sequence
            slice_idx - The slice index to use to extract a slice from the 3D image array (and other arrays).
                        If None, the entire 3D array is used.
        """
        lesion_vol = lesion_load(cc_bin, self.spacing)  # Lesion volume (or lesion-slice volume) (float, mm^3)
        total_load = lesion_load(cc_arr, self.spacing)  # Total lesion load for full volume  (float, mm^3)
        position_stats = self.get_position_stats(cc_bin, slice_idx)  # dict
        # Extract relevant statistics from overlapping component in adjacent slices, if applicable
        if self.process_slicewise:
            # Probs for current lesion (or lesion-slice) (dict)
            lesion_probs = summarise_probabilities(pmaps_arr[slice_idx], cc_bin)
            adjacent_stats = analyse_adjacent_slices(pmaps_arr, cc_arr, slice_num=slice_idx, spacing=self.spacing)
            slice_load = lesion_load(cc_arr[slice_idx], self.spacing)  # Total lesion load for slice (float, mm^3)
        else:
            lesion_probs = summarise_probabilities(pmaps_arr, cc_bin)
            adjacent_stats = {}

        general_features = {
            **position_stats,
            **adjacent_stats
        }
        seq_specific_features = {
            seq_name: {
                'lesion_vol': lesion_vol,
                'total_load': total_load,
                'iou': 1.0,  # iou of pred with itself, just for consistency, because we will have iou with other seqs
                **lesion_probs
            }
        }
        if self.process_slicewise:
            seq_specific_features[seq_name]['slice_load'] = slice_load
        return general_features, seq_specific_features

    def process_other_seq_2d_or_3d(self, other_pmaps_arr: np.ndarray, current_cc_arr: np.ndarray,
                                   other_seq_name: str, other_thresh: float, slice_idx: Optional[int] = None):
        """ Process the other connected component and extract features for the current slice or the 3D component.
        Args:
            other_pmaps_arr (np.ndarray): The prediction array for the other sequence
            current_cc_arr (np.ndarray): The connected component array for the current sequence, with only the single
                                         connected component of interest (if processing slicewise, then this is a slice)
            other_seq_name (str): The name of the other sequence to be processed, corresponding to the pmaps_arr
            other_thresh (float): The threshold to be used to load the connected component for the other sequence
            slice_idx (int): The slice index to use to extract a slice from the 3D image array (and other arrays).
        """
        # other_probs, other_slice/total_load, other_lesion_vol, other_iou
        # Load the corresponding connected component for the other sequence
        other_cc_arr = self.load_and_extract_ims(self.cc_dir / f'thr-{other_thresh}',
                                                 filenames=self.cc_filenames,
                                                 filename_patterns=[other_seq_name],
                                                 interpolator=sitk.sitkNearestNeighbor
                                                 )[0][0]
        # Get the corresponding field of view (FOV) mask for the other sequence
        fov_arr_names = self.load_and_extract_ims(self.pred_dir,
                                                  self.fov_filenames,
                                                  [other_seq_name + '_FOV'],
                                                  interpolator=sitk.sitkNearestNeighbor
                                                  )
        if len(fov_arr_names) > 0:
            fov_arr = fov_arr_names[0][0]
            in_fov_pct = fov_arr[current_cc_arr > 0].sum() / current_cc_arr.sum()
            if in_fov_pct < 0.25:
                # If the current connected component is not in the FOV of the other sequence,
                #   then return NAN features rather than zeros so that we keep this information
                feat_names = ['total_load', 'lesion_vol', 'iou', 'max', 'min', 'mean',
                              'median', '10th', '20th', '90th', '95th', '99th']
                features = {
                    other_seq_name: {feat: np.nan for feat in feat_names}
                }
                if self.process_slicewise:
                    features[other_seq_name]['slice_load'] = np.nan
                return features

        total_load = lesion_load(other_cc_arr, self.spacing)  # Total lesion load (or slice load)  (float, mm^3)

        if self.process_slicewise:
            other_cc_arr = other_cc_arr[slice_idx]
            other_pmaps_arr = other_pmaps_arr[slice_idx]

        # Probs for the current lesion (or lesion-slice) in the paired sequence
        lesion_probs = summarise_probabilities(other_pmaps_arr, current_cc_arr)  # dict
        # Get any predictions in the other sequence that overlap with the current connected component
        other_ccs = other_cc_arr[current_cc_arr > 0]
        if other_ccs.sum() > 0:
            other_cc_ids = np.unique(other_ccs)[1:]
            # Get the lesion volume for overlapping predictions (note this might mean multiple lesions in the other seq)
            other_lesion_vol = lesion_load(np.isin(other_cc_arr, other_cc_ids), self.spacing)
            # Get the IoU for the overlapping predictions (again, this might mean multiple lesions)
            other_iou = get_iou(current_cc_arr, np.isin(other_cc_arr, other_cc_ids))
        else:
            other_lesion_vol = 0
            other_iou = 0

        features = {
            other_seq_name: {
                'total_load': total_load,
                'lesion_vol': other_lesion_vol,
                'iou': other_iou,
                **lesion_probs
            }
        }
        if self.process_slicewise:
            features[other_seq_name]['slice_load'] = lesion_load(other_cc_arr, self.spacing)

        return features

    def gather_features(self) -> pd.DataFrame:
        # Load the pmaps (and their corresponding sequence identifier)
        pmaps_arrs_names = self.load_and_extract_ims(self.pred_dir, self.pred_filenames, self.seqnames)

        data = []
        for pmaps_arr, pmaps_name in pmaps_arrs_names:
            for cc_arr, cc_name, cc_ids, thresh in self.loop_ccs():
                for cc_id in cc_ids:
                    cc_bin = (cc_arr == cc_id).astype(np.uint8)
                    if self.process_slicewise:
                        for i_x, cc_slice in enumerate(cc_bin):
                            if np.sum(cc_slice) == 0:
                                continue
                            if pmaps_name == cc_name:
                                # Process the features for the source sequence
                                general_features, seq_features = self.process_current_2d_or_3d(
                                    pmaps_arr, cc_slice, cc_arr, cc_name, slice_idx=i_x)
                            else:
                                # Process the features for the paired sequence(s)
                                seq_features = self.process_other_seq_2d_or_3d(
                                    pmaps_arr, cc_slice, pmaps_name, thresh, slice_idx=i_x)
                                general_features = {}

                            row = self.update_row(cc_id, cc_name, thresh, slice_idx=i_x,
                                                  seq_features=seq_features, general_features=general_features)
                            data.append(row)

                    else:
                        # Process whole 3D connected components
                        if pmaps_name == cc_name:
                            # Process the features for the source sequence
                            general_features, seq_features = self.process_current_2d_or_3d(pmaps_arr, cc_bin, cc_arr,
                                                                                           cc_name)
                        else:
                            # Process the features for the paired sequence(s)
                            seq_features = self.process_other_seq_2d_or_3d(pmaps_arr, cc_bin, pmaps_name, thresh)
                            general_features = {}

                        row = self.update_row(cc_id, cc_name, thresh, seq_features=seq_features,
                                              general_features=general_features)
                        data.append(row)

        df = pd.DataFrame(data)
        # take the first non-null value for each column
        id_cols = ['case_id', 'cc_id', 'seq', 'threshold']
        if self.process_slicewise:
            id_cols.append('slice_num')
        combined_df = df.groupby(id_cols).agg('first').reset_index()

        return combined_df

    def process_single_subject(self) -> pd.DataFrame:
        """ Process a single subject and extract features for all sequences and thresholds.
        Returns:
            pd.DataFrame: A dataframe containing the features for the subject.
        """
        # Load the ground truth image
        self.update_anat_im()

        # Load the SC segmentation
        sc_seg_arr = load_reorient(self.pred_dir / 'T2_sc_seg.nii.gz', 'LAS')
        sc_seg_arr = sitk_to_numpy(resample_to_ref(sc_seg_arr, self.anat_im, interpolator=sitk.sitkNearestNeighbor))
        # Update the spinal cord information
        self.update_sc_info(sc_seg_arr)

        # Gather probs, position, volumes, and IoU features
        main_features_df = self.gather_features()
        # Gather intensity statistics for the current subject
        #   This is separated from the above because only these features require the anat ims & SC seg
        intensity_stats_df = self.gather_intensity_stats(sc_seg_arr)

        # Merge the two dataframes
        join_cols = ['case_id', 'cc_id', 'seq', 'threshold']
        if self.process_slicewise:
            join_cols.append('slice_num')

        features_df = pd.merge(main_features_df, intensity_stats_df, how='inner', on=join_cols)
        # Check that the lengths match
        if len(features_df) != len(main_features_df) or len(features_df) != len(intensity_stats_df):
            raise ValueError("The lengths of the dataframes do not match after merging.")

        # Add the metadata (fixed values for a single subject)
        for col, val in self.metadata_sub.items():
            features_df[col] = val

        return features_df

    def save_features(self, features_df):
        if features_df is not None:
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(self.out_file, index=False)
        else:
            raise ValueError("No features dataframe to save.")


def main(args):
    pred_names = ['T2_pred_sc-masked.nii.gz',
                  'MP2RAGE_pred_resampled_sc-masked.nii.gz',
                  'PSIR_pred_warped_sc-masked.nii.gz',
                  'STIR_pred_warped_sc-masked.nii.gz',
                  'preds_mean.nii.gz']

    fov_names = ['MP2RAGE_FOV_resampled.nii.gz', 'PSIR_FOV_resampled.nii.gz', 'STIR_FOV_resampled.nii.gz']

    cc_names = [f'{pred_name.replace(".nii.gz", "")}_cc.nii.gz' for pred_name in pred_names]

    generate_ccs(args.pred_dir / 'tmp', args.thresholds, pred_names, cc_names)
    # Clean up connected components for low thresholds - remove the new (thousands of) CCs introduced at very low thresh
    clean_up_ccs(args.pred_dir / 'tmp', cc_names, [0.000001, 0.00001], reference_threshold=0.0001)
    if not args.process_slicewise:
        generate_neighbourhoods(args.pred_dir / 'tmp', cc_names)

    metadata_df = pd.read_csv(args.metadata_path, sep='\t')

    processor = FeatureProcessor(
        process_slicewise=args.process_slicewise,
        pred_dir_parent=args.pred_dir,
        anat_dir=args.anat_dir,
        thresholds=args.thresholds,
        pred_filenames=pred_names,
        cc_filenames=cc_names,
        fov_filenames=fov_names,
        metadata_df=metadata_df,
        metadata_cols=['coverage', 'device_manufacturer_norm', 'device_field_strength'],
    )
    features_df = processor.process_single_subject()
    processor.save_features(features_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', '-p', type=Path, required=True,
                        help='Path to the directory containing the predictions')
    parser.add_argument('--anat_dir', '-a', type=Path, required=True,
                        help='Path to the directory containing the raw anatomical images')
    # TODO: restrict to chosen thresholds ?
    parser.add_argument('--thresholds', '-t', type=float, nargs='+',
                        default=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5],
                        help='Thresholds to use for generating connected components.')
    parser.add_argument('--metadata_path', '-m', type=Path, help='Path to TSV file containing upper/lower info',
                        required=True)
    parser.add_argument('--process_slicewise', '-ps', action='store_true', help='Process features slice-wise')

    input_args = parser.parse_args()

    main(input_args)
