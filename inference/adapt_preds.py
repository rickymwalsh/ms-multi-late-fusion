
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np
import sklearn.ensemble
import xgboost as xgb
from tqdm import tqdm
from skimage.measure import label
from typing import List, Dict, Tuple, Optional
from warnings import warn
import argparse
import joblib

from feature_engineering import load_reorient
from postprocess_preds import check_exists
from im_utils import sitk_to_numpy, new_image_from_ref


def prepare_features(data_path):
    features_df = pd.read_csv(data_path)
    # Drop the ground-truth IoU column if it exists, as it should not be used for predictions.
    if 'iou_gt' in features_df.columns:
        features_df = features_df.drop(columns=['iou_gt'])

    # Rename seq to mask, because it just refers to the current mask we're using for the current CC
    features_df = features_df.rename(columns={'seq': 'mask'})
    # Get other_seq based on which column has non-null values
    def check_col(row, seq, df):
        """ Helper function to check if a column has non-null values. """
        return pd.notnull(row[f'{seq}_total_load']) if f'{seq}_total_load' in df.columns else False

    features_df['other_seq'] = features_df.apply(
        lambda row: 'MP2RAGE' if check_col('MP2RAGE', row, features_df) else
                    ('STIR' if check_col('STIR', row, features_df) else
                     ('PSIR' if check_col('PSIR', row, features_df) else None)),
        axis=1)

    def get_vol(seq, else_cond):
        """ Helper function to get the lesion volume for a given sequence. """
        vol_col = features_df[f'{seq}_lesion_vol'] if f'{seq}_lesion_vol' in features_df.columns else np.nan
        return np.where(features_df['mask'] == seq, vol_col, else_cond)

    features_df['current_vol'] = get_vol('T2',
                                         get_vol('STIR',
                                                 get_vol('MP2RAGE',
                                                         get_vol('PSIR',
                                                                 features_df['mean_lesion_vol']))))

    # Create one-hot columns (do it this way to ensure all combinations are present, even if not in current DF)
    cat_cols = ['coverage', 'device_manufacturer_norm', 'mask', 'other_seq']
    for col, values in zip(cat_cols,
                           [['UpperSC', 'LowerSC'], ['Philips', 'Siemens'], ['MP2RAGE', 'PSIR', 'STIR', 'T2', 'mean'],
                            ['MP2RAGE', 'PSIR', 'STIR', 'T2']]):
        for value in values:
            features_df[f'{col}_{value}'] = np.where(features_df[col] == value, 1, 0)
    features_df = features_df.drop(columns=cat_cols)

    # Convert inf to nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    # Drop ID columns
    ids_cols = ['case_id', 'cc_id']
    if 'slice_num' in features_df.columns:
        ids_cols += ['slice_num']
    ids_labels = features_df[ids_cols]
    features_df = features_df.drop(columns=ids_cols)

    # DECORRELATED features!
    # Drop columns which end with any of the following because they are highly correlated with other features
    drop_cols = [col for col in features_df.columns if col.endswith(
        ('_95th_perc_intensity', '_median_intensity', '_contrast_vs_sc',
         '_99th', '_95th', '_90th', '_mean', '_20th', '_min'))]

    drop_cols.pop(drop_cols.index('mask_mean'))  # Keep mask_mean -> just indicates the current mask is the mean mask
    features_df = features_df.drop(columns=drop_cols)

    # Create single column for each of the sequence-specific columns to avoid NAs for training the models
    for col_suffix in ['_total_load', '_lesion_vol', '_iou', '_max', '_median', '_10th', '_mean_intensity',
                       '_min_intensity', '_max_intensity', '_5th_perc_intensity', '_mean_vs_sc_std',
                       '_contrast_vs_neighbourhood']:
        cols_to_use = [f'{seq}{col_suffix}' for seq in ['MP2RAGE', 'STIR', 'PSIR']
                       if f'{seq}{col_suffix}' in features_df.columns]
        # For MP2RAGE + STIR cases, take the MP2RAGE values first. (For predicted CCs outside of MP2RAGE FOV, the
        # features will be NaN, so we will use the STIR values anyway.)
        cols_to_use = sorted(cols_to_use, key=lambda x: x.startswith('MP2RAGE'), reverse=True)  # MP2RAGE first
        features_df[f'other{col_suffix}'] = features_df[cols_to_use].bfill(axis=1).iloc[:, 0]  # bfill gets the first non-NA value
        features_df = features_df.drop(columns=cols_to_use)

    ids_labels['threshold'] = features_df['threshold']
    mask_cols = [col for col in features_df.columns if col.startswith('mask_')]
    ids_labels['mask'] = features_df[mask_cols].idxmax(axis=1).str.replace('mask_', '')

    return features_df, ids_labels


def get_preds(features_df: pd.DataFrame, ids: pd.DataFrame, model_dir: Path,
              test_fold_df: Optional[pd.DataFrame] = None
              ) -> pd.DataFrame:
    """
    Run the given features through the model(s) to get predicted probabilities. If there are multiple models in the
    directory, then the average of all predicted probabilities for a given case will be taken.
    Args:
        features_df:    DataFrame with samples in rows and features in columns to be used for predictions.
        ids:            DataFrame with the ID columns used to uniquely identify the rows in feature_df, e.g. with
                        case_id, cc_id, etc.
        model_dir:      Path to the directory with the model(s) to use for predictions. Models should be saved
                        as .joblib files.
        test_fold_df:   Optional DataFrame with the test fold information. If provided, will be used to filter the
                        features_df to only include the rows that are in the test fold. This is used for
                        cross-validation scenarios where the model was trained on a subset of the data.
    Return:
        preds_df:   DataFrame with one row per row in the features_df, retaining the ID cols
                    and the predicted probabilities for each connected component.
    """
    model_files = list(model_dir.glob('*.joblib'))
    if not model_files:
        raise ValueError(f'No models found in {model_dir}. Please provide a valid directory with models.')

    preds_df = pd.DataFrame()
    for i, model_file in enumerate(model_files):
        model = joblib.load(model_file)
        # Ensure column order is same as what the model was trained on
        if isinstance(model, xgb.XGBClassifier):
            col_order = model.get_booster().feature_names
        elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
            col_order = model.feature_names_in_.tolist()
        features_df = features_df[col_order]

        if test_fold_df is None:
            preds = model.predict_proba(features_df)[:, 1]  # Get prob of positive class
            preds_df[f'model_{i}'] = preds  # Store predictions from each model in a separate column
        else:
            # Model filename should be fold-[0-5].joblib
            fold = int(model_file.stem.split('-')[-1])
            case_ids = test_fold_df[test_fold_df['test_fold'] == fold]['case_id'].unique()
            features_subset = features_df[ids['case_id'].isin(case_ids)]
            ids_subset = ids[ids['case_id'].isin(case_ids)].set_index(ids.columns.tolist())
            if features_subset.empty or len(ids_subset) == 0:
                continue
            preds = model.predict_proba(features_subset)[:, 1]
            preds_df_tmp = pd.DataFrame({f'model_{i}': preds}, index=ids_subset.index)
            preds_df = pd.concat([preds_df, preds_df_tmp], axis=0)

    preds_df['y_probs'] = preds_df.mean(axis=1)  # Average the probabilities across all models

    if test_fold_df is None:
        preds_df = pd.concat([ids, preds_df], axis=1)
    else:
        preds_df = preds_df.reset_index()

    return preds_df


class Predaptor:
    def __init__(self, cc_dir: Path, out_dir: Path, anat_dir: Path, preds_df_path: Optional[Path] = None,
                 preds_df: Optional[pd.DataFrame] = None, id_cols: List[str] = None,
                 masks_to_use: Optional[Dict] = None, thresh_to_use: Optional[Dict] = None
                 ):
        to_check = [cc_dir, anat_dir] if preds_df_path is None else [cc_dir, anat_dir, preds_df_path]
        check_exists(to_check, error=True)
        assert (preds_df is None) != (preds_df_path is None), 'Either preds_df or preds_df_path must be provided.'

        self.preds_df = pd.read_csv(str(preds_df_path)) if preds_df_path is not None else preds_df
        self.id_cols = id_cols if id_cols is not None else ['case_id', 'cc_id', 'slice_num', 'threshold']
        self.sense_check_preds()

        self.cc_dir = cc_dir
        self.out_dir = out_dir  # TODO: make output behaviour consistent for both simple/full adaptation
        self.anat_dir = anat_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.process_slicewise = True

        if masks_to_use is not None:
            self.masks_to_use = masks_to_use
        else:
            self.masks_to_use = {'STIR': 'T2_pred_sc-masked_cc.nii.gz',
                                 'PSIR': 'T2_pred_sc-masked_cc.nii.gz',
                                 'MP2RAGE': 'preds_mean_cc.nii.gz',
                                 }
        if thresh_to_use is not None:
            self.thresh_to_use = thresh_to_use
        else:
            self.thresh_to_use = {'STIR': 0.00001, 'PSIR': 0.000001, 'MP2RAGE': 0.0001}

        self.target_orientation = self.get_orientation()
        self.subject_group = self.get_subject_group()

    def sense_check_preds(self):
        """ Check that preds_df has the expected columns."""
        if not all(col in self.preds_df.columns for col in self.id_cols + ['y_probs']):
            raise ValueError(f'The preds_df DataFrame must have the columns: {self.id_cols + ["y_probs"]}')
        # Check unique row per case_id, cc_id, slice_num
        if not self.preds_df.groupby(self.id_cols + ['mask']).size().eq(1).all():
            # If the only difference is the y_probs value, then just take the max y_probs for now
            max_prob = self.preds_df.groupby(self.id_cols + ['mask'])['y_probs'].max().reset_index()
            self.preds_df = pd.merge(self.preds_df[~self.preds_df.duplicated()], max_prob,
                                     on=self.id_cols + ['mask'] + ['y_probs'], how='inner')
            print('Dups found in preds_df. Taking max y_probs for each case_id, cc_id, slice_num, threshold, mask, etc.')
            # raise ValueError('There must be a unique row per case_id, cc_id, threshold, mask, etc.'
            #                  ' in the preds_df DataFrame')

    def get_orientation(self) -> Dict:
        """ Get the orientation of the GT images for each subject/case.
        Returns:
            orientations (Dict): A dictionary with the subject/case IDs as keys and the SITK orientation strings as values.
        """
        t2_path = list(self.anat_dir.glob('*T2.nii.gz'))[0]
        t2_im = sitk.ReadImage(t2_path)
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(t2_im.GetDirection())
        return orientation

    def get_subject_group(self) -> str:
        """ Check which supplementary sequences are available for the subject. """
        anat_files = list(self.anat_dir.glob('*.nii.gz'))

        group = ''
        for seq in ['STIR', 'PSIR', 'MP2RAGE']:
            if any(seq in str(f) for f in anat_files):
                if group:
                    group += '_'
                group += seq
        if not group:
            warn('No supplementary sequences found in the anat_dir. Using only T2 sequence.')
            group = 'T2'

        return group

    def get_fov_ims(self, seq: str) -> np.ndarray:
        """ Get the field of view image for a given sequence.
        Args:
            seq (str): The sequence to get the FOV image for. E.g. 'T2', 'STIR', 'PSIR', 'MP2RAGE'.
        Returns:
            fov_im (np.ndarray): The 3D array of the FOV image for the given sequence.
        """
        fov_path = self.out_dir / 'tmp' / f'{seq}_FOV_resampled.nii.gz'
        if not fov_path.exists():
            raise FileNotFoundError(f'FOV image for {seq} not found at {fov_path}.')
        fov_im = load_reorient(fov_path, orientation='LAS')  # To be consistent with the loaded CC arr
        fov_im = sitk_to_numpy(fov_im)
        return fov_im

    @staticmethod
    def get_old_new_cc_matches(old_cc_arr, new_cc_arr):
        """ Get the corresponding IDs for the old and new connected components.
        Args:
            old_cc_arr (np.ndarray): The 3D array of the old connected components.
            new_cc_arr (np.ndarray): The 3D array of the new connected components.
        Returns:
            new_ccs_data (pd.DataFrame): A DataFrame with columns: old_cc_id, new_cc_id, and slice_num
        """
        new_ccs_data = []
        new_ids = np.unique(new_cc_arr)
        if len(new_ids) > 1:
            for new_id in new_ids[1:]:
                old_ids = np.unique(old_cc_arr[new_cc_arr == new_id])
                for slice_num in np.where(new_cc_arr == new_id)[0]:
                    if len(old_ids) == 1 and old_ids[0] == 0:
                        new_ccs_data.append([None, new_id, slice_num])
                    else:
                        for old_id in old_ids:
                            if old_id != 0:
                                new_ccs_data.append([old_id, new_id, slice_num])

        return pd.DataFrame(new_ccs_data, columns=['old_cc_id', 'new_cc_id', 'slice_num'])

    def get_all_overlapping_lesions(self, cc_arrs: List[np.ndarray], arr_ids: List[List], arr_id_names: List[str]):
        """ Get all possible combinations of overlapping connected components SLICE-WISE with a brute-force approach.
        Args:
            cc_arrs (List): List of 3D arrays of connected components.
            arr_ids (List): List of lists of IDs for each connected component array. Length must be the same as cc_arrs.
            arr_id_names (List): List of names for the IDs in arr_ids. Length must be the same as the sublist length in arr_ids.
        Returns:
            overlapping_lesions_df (pd.DataFrame): A DataFrame with columns: cc_id_1, cc_id_2, slice_num, and the
                                variables from arr_ids (e.g. thresh1, thresh2, seq1, seq2)
        """
        if len(cc_arrs) != len(arr_ids):
            raise ValueError('Length of cc_arrs and arr_ids must be the same.')
        if all(cc_arrs[0].shape != cc_arr.shape for cc_arr in cc_arrs):
            raise ValueError('All connected component arrays must have the same shape.')

        print('Getting overlapping lesions...')

        overlapping_lesions = []
        processed_cc_ids = set()
        for i, cc_arr1 in tqdm(enumerate(cc_arrs), total=len(cc_arrs)):
            cc_ids1 = np.unique(cc_arr1)
            if len(cc_ids1) == 1 and cc_ids1[0] == 0:
                continue
            for j, cc_arr2 in enumerate(cc_arrs):
                if j <= i:
                    continue
                cc_arr2_masked = cc_arr2 * (cc_arr1 != 0)
                if cc_arr2_masked.sum() == 0:
                    continue

                cc_ids2 = np.unique(cc_arr2_masked)
                for cc_id2 in cc_ids2[1:]:
                    if self.process_slicewise:
                        for slice_num in np.where(cc_arr2_masked == cc_id2)[0]:
                            for cc_id1 in np.unique(cc_arr1[slice_num][cc_arr2_masked[slice_num] == cc_id2]):
                                if cc_id1 == 0:
                                    continue
                                overlapping_lesions.append([cc_id1, cc_id2, slice_num, *arr_ids[i], *arr_ids[j]])
                                processed_cc_ids.update([(cc_id1, slice_num, *arr_ids[i]), (cc_id2, slice_num, *arr_ids[j])])
                    else:
                        for cc_id1 in np.unique(cc_arr1[cc_arr2_masked == cc_id2]):
                            if cc_id1 == 0:
                                continue
                            overlapping_lesions.append([cc_id1, cc_id2, *arr_ids[i], *arr_ids[j]])
                            processed_cc_ids.update([(cc_id1, *arr_ids[i]), (cc_id2, *arr_ids[j])])
            # Add any CC ID that doesn't overlap with predictions from the other seqs
            if self.process_slicewise:
                for slice_num in np.where(cc_arr1 != 0)[0]:
                    for cc_id in np.unique(cc_arr1[slice_num])[1:]:
                        if (cc_id, slice_num, *arr_ids[i]) not in processed_cc_ids:
                            overlapping_lesions.append([cc_id, None, slice_num, *arr_ids[i], *[None]*len(arr_ids[i])])
            else:
                for cc_id in cc_ids1[1:]:
                    if (cc_id, *arr_ids[i]) not in processed_cc_ids:
                        overlapping_lesions.append([cc_id, None, *arr_ids[i], *[None]*len(arr_ids[i])])

        base_cols = ['cc_id_1', 'cc_id_2', 'slice_num'] if self.process_slicewise else ['cc_id_1', 'cc_id_2']
        colnames = base_cols + [f'{name}1' for name in arr_id_names] + [f'{name}2' for name in arr_id_names]
        overlapping_lesions_df = pd.DataFrame(overlapping_lesions, columns=colnames)

        return overlapping_lesions_df

    def structure_overlap_dict(self, overlapping_lesions_df: pd.DataFrame, probs_df: pd.DataFrame, id_names: List[str]):
        """ Structure the overlapping lesions as a dictionary for easier processing.
        Args:
            overlapping_lesions_df (pd.DataFrame): A DataFrame with columns: cc_id_1, cc_id_2, slice_num, and
                    identifying variables for where cc_id_1 and cc_id_2 come from (e.g. thresh1, thresh2, seq1, seq2)
            probs_df (pd.DataFrame): A DataFrame with the probabilities for each connected component.
                Should have columns: cc_id, slice_num, predicted_prob, and the matching identifying variables as in
                overlapping_lesions_df.
            id_names (List): The base names of the identifying variables in the DataFrame. E.g. ['thresh', 'mask']
        Returns:
            overlapping_lesions_dict (Dict): A dictionary with the keys as the combination of the ID columns in
                overlapping_lesions_df, along with the cc_id and slice_num.
                E.g. for CC 5 from seq T2 at threshold 0.0001 at slice 3 in the image, the key would be
                ('0.0001', 'T2', 3, 5). The order here is determined by the order of the ID columns in the input df.
                The values are another dict, containing the {'prob': prob, overlaps: [...]}
                E.g.
                    {
                        ('0.0001', 'T2', 3, 5): {
                            'prob': 0.8,
                            'overlaps': [
                                ('0.001', 'T2', 3, 1),
                                ('0.0001', 'STIR', 3, 2)
                            ]
                        },
                        ...
                    }
        """
        # Add the probabilities for each of the overlapping lesions.
        maybe_slice = ['slice_num'] if self.process_slicewise else []

        # Ensure any numerical columns have the same type for merging (could be changed because of missing values)
        cols_to_check = ['cc_id_1', 'cc_id_2'] + [f'{name}1' for name in id_names] + [f'{name}2' for name in id_names]
        for col in cols_to_check:
            other_col_name = col.replace('_1', '').replace('_2', '').replace('1', '').replace('2', '')
            if col in overlapping_lesions_df.columns and other_col_name in probs_df.columns:
                if probs_df[other_col_name].dtype != overlapping_lesions_df[col].dtype:
                    # Convert to the same type after replacing NA with 0 (since integer types cannot have NA)
                    overlapping_lesions_df[col] = overlapping_lesions_df[col].fillna(0).astype(probs_df[other_col_name].dtype)

        df = pd.merge(overlapping_lesions_df, probs_df.rename(columns={'predicted_prob': 'prob1'}),
                      left_on=['cc_id_1'] + maybe_slice + [f'{name}1' for name in id_names],
                      right_on=['cc_id'] + maybe_slice + id_names, how='left') \
            .merge(probs_df.rename(columns={'predicted_prob': 'prob2'}),
                   left_on=['cc_id_2'] + maybe_slice + [f'{name}2' for name in id_names],
                   right_on=['cc_id'] + maybe_slice + id_names, how='left')

        overlapping_lesions_dict = {}
        for _, row in df.iterrows():
            key1 = tuple(row[[f'{name}1' for name in id_names] + maybe_slice + ['cc_id_1']])
            key2 = tuple(row[[f'{name}2' for name in id_names] + maybe_slice + ['cc_id_2']])

            if key1 not in overlapping_lesions_dict:
                overlapping_lesions_dict[key1] = {'prob': row['prob1'], 'overlaps': []}
            if key2 not in overlapping_lesions_dict:
                overlapping_lesions_dict[key2] = {'prob': row['prob2'], 'overlaps': []}

            overlapping_lesions_dict[key1]['overlaps'].append(key2)
            overlapping_lesions_dict[key2]['overlaps'].append(key1)

        return overlapping_lesions_dict

    @staticmethod
    def get_lesions_to_keep(overlapping_lesions_dict: Dict):
        """ Iterate over all overlaps and take the lesion with the highest probability until there are no more overlaps.
        Args:
            overlapping_lesions_dict (Dict): A dictionary with the CC IDs, probabilities and overlap with other CC IDs.
            E.g. {('0.5', 'T2', 3, 5): {'prob': 0.8, 'overlaps': [('0.5', 'STIR', 3, 2), ('0.001', 'T2', 3, 1)]}, ...}
            where the keys here uniquely identify a connected component (thresh, seq, slice_number, cc_id)
        Returns:
            lesions_to_keep (Dict): A subset of the input overlapping_lesions_dict, containing only the lesions to keep.
        """
        # Get sorted keys based on probability
        cc_keys = sorted(overlapping_lesions_dict.keys(), key=lambda x: overlapping_lesions_dict[x]['prob'], reverse=True)
        lesions_to_keep = {}
        for key in cc_keys:
            if len(overlapping_lesions_dict) == 0:
                break
            if key in overlapping_lesions_dict:
                overlaps = overlapping_lesions_dict[key]['overlaps']
                lesions_to_keep[key] = overlapping_lesions_dict.pop(key)
                for overlap_key in overlaps:
                    overlapping_lesions_dict.pop(overlap_key, None)

        return lesions_to_keep

    def create_new_seg_mask(self, cc_arrs: List[np.ndarray], im_ids: List[Tuple], lesions_to_keep: Dict
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """ Create a new segmentation mask with the connected components to keep.
        Args:
            cc_arrs (List): List of 3D arrays of connected components.
            im_ids (List): List of tuples of identifying variables for each connected component array. These should
                correspond to the keys in the lesions_to_keep dictionary, without the last two tuple elements (slice_num, cc_id).
            lesions_to_keep (Dict): A dictionary with the connected components to keep and their probabilities.
                E.g. {('0.5', 'T2', 9, 5): {'prob': 0.8}, ...} -> keep CC num 5 in slice 9 from the T2 image with threshold 0.5
        Returns:
            seg (np.ndarray): The new binary segmentation mask.
            probs (np.ndarray): The probabilities for each lesion-slice that was kept.
        """
        if len(cc_arrs) != len(im_ids):
            raise ValueError('Length of cc_arrs and im_ids must be the same.')
        if all(cc_arrs[0].shape != cc_arr.shape for cc_arr in cc_arrs):
            raise ValueError('All connected component arrays must have the same shape.')

        seg = np.zeros_like(cc_arrs[0])
        probs = np.zeros_like(seg, dtype=np.float32)

        for cc_global_id, data in lesions_to_keep.items():
            if self.process_slicewise:
                # Split the tuple of IDs into IDs to identify the image and IDs to identify the lesion in the image
                slice_num, cc_num = cc_global_id[-2:]
                im_id = cc_global_id[:-2]
                # Get which array this connected component is from
                im_idx = im_ids.index(im_id)
                # Identify the pixels for this connected component
                loc = np.where(cc_arrs[im_idx][slice_num] == cc_num)  # Relies on slice axis being first axis in the image.
                # Set the segmentation mask and probabilities
                seg[slice_num][loc] = 1
                probs[slice_num][loc] = data['prob']
            else:
                cc_num = cc_global_id[-1]  # Extract the connected component ID
                im_id = cc_global_id[:-1]  # Extract the image ID(s)
                im_idx = im_ids.index(im_id)  # Get which array this connected component is from
                seg[cc_arrs[im_idx] == cc_num] = 1  # Set the segmentation mask
                probs[cc_arrs[im_idx] == cc_num] = data['prob']  # Set the probabilities

        return seg, probs

    @staticmethod
    def reprocess_ccs(arr: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Re-compute the 3D connected components for one image, and assign the max probability to each connected component.
        Args:
            arr (np.ndarray): The 3D array of existing connected components or binary seg.
            probs (np.ndarray): The probabilities for each connected component or voxel. (Same shape as arr)
        Returns:
            ccs (np.ndarray): The new 3D connected components.
            probs (np.ndarray): The max probability for each connected component. (Same shape as ccs)
        """
        ccs = label(arr > 0)

        cc_ids = np.unique(ccs)
        if len(cc_ids) == 1 and cc_ids[0] == 0:
            # No connected components
            return ccs, probs

        # Get the max probability for each connected component
        for cc_id in cc_ids[1:]:
            probs[ccs == cc_id] = probs[ccs == cc_id].max()

        return ccs, probs

    @staticmethod
    def suppress_low_prob_slices(cc_arr: np.ndarray, probs: np.ndarray, min_prob: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Suppress slices in connected components with low probabilities.
        Looks at the 3D connected component, and if the max CC probability < min_prob, then the CC is kept intact!
        Otherwise, if the max CC probability >= min_prob, then slices with probabilities below min_prob are removed.
        Args:
            cc_arr (np.ndarray): The 3D array of connected components.
            probs (np.ndarray): The probabilities for each connected component. (Same shape as cc_arr)
            min_prob (float): The minimum probability to keep a lesion on a slice.
        Returns:
            cc_arr (np.ndarray): The updated 3D array of connected components.
            probs (np.ndarray): The updated probabilities for each connected component. (Same shape as cc_arr)
        """
        cc_ids = np.unique(cc_arr)
        if len(cc_ids) == 1 and cc_ids[0] == 0:
            # No connected components
            return cc_arr, probs

        for cc_id in cc_ids[1:]:
            # If all the probs for this 3D connected component are below min_prob, then keep it intact
            if probs[cc_arr == cc_id].max() < min_prob:
                continue
            # Otherwise, we remove any slices from the connected component with probabilities below min_prob
            for slice_num in np.unique(np.where(cc_arr == cc_id)[0]):
                is_this_cc = cc_arr[slice_num] == cc_id
                if probs[slice_num][is_this_cc].max() < min_prob:
                    cc_arr[slice_num][is_this_cc] = 0
                    probs[slice_num][is_this_cc] = 0

        return cc_arr, probs

    @staticmethod
    def get_probs_from_ccs(ccs: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
        """ Get the max probability for each connected component, in table form.
        Args:
            ccs (np.ndarray): The 3D array of connected components.
            probs (np.ndarray): The probabilities for each connected component. (Same shape as ccs)
        Returns:
            probs_df (pd.DataFrame): A DataFrame with columns: label, p  -> label is the unique CC ID, p is the max prob
        """
        if ccs.shape != probs.shape:
            raise ValueError('ccs and probs must have the same shape.')

        cc_ids = np.unique(ccs)
        if len(cc_ids) == 1 and cc_ids[0] == 0:
            # No connected components -> will save empty table
            probs_df = pd.DataFrame(columns=['label', 'p'])
        else:
            probs_data = []
            for cc_id in cc_ids[1:]:
                probs_data.append([cc_id, probs[ccs == cc_id].max()])
            probs_df = pd.DataFrame(probs_data, columns=['label', 'p'])

        return probs_df

    def save_outputs(self, cc_arr: np.ndarray, probs_arr: np.ndarray, cc_im: sitk.Image, out_dir: Path):
        """ Save the outputs of the adaptation as "prediction_proba.csv" and "prediction_map.nii.gz"
        Args:
            cc_arr (np.ndarray): The 3D array of connected components.
            probs_arr (np.ndarray): The probabilities for each connected component. (Same shape as cc_arr)
            cc_im (sitk.Image): The original connected component image.
            out_dir (Path): The output directory for a given case to save the instances and probabilities.
        Returns:
            None
        """
        # Save the new connected components
        cc_im = new_image_from_ref(cc_arr, cc_im)
        cc_im = sitk.DICOMOrient(cc_im, self.target_orientation)
        sitk.WriteImage(cc_im, out_dir / 'prediction_map.nii.gz')

        # Extract the probability for each connected component
        probs_df = self.get_probs_from_ccs(cc_arr, probs_arr)
        probs_df.to_csv(out_dir / 'prediction_proba.csv', index=False)

    def process_cc_probs_simple(self, preds_rows: pd.DataFrame, cc_arr: np.ndarray, probs_arr: np.ndarray,
                                min_prob: Optional[float] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
        for _, row in preds_rows.iterrows():
            cc_id = row['cc_id']
            slice_num = row['slice_num']
            y_prob = row['y_probs']
            max_cc_prob = preds_rows[preds_rows['cc_id'] == cc_id]['y_probs'].max()
            # Suppress lesion-slice if low prob and at least one slice in 3D CC has higher prob
            if y_prob < min_prob and max_cc_prob >= min_prob:
                cc_arr[slice_num][cc_arr[slice_num] == cc_id] = 0
            else:
                # Set the probability for this slice
                probs_arr[slice_num][cc_arr[slice_num] == cc_id] = y_prob

        # Recalculate the connected components and their associated max probabilities
        cc_arr, probs_arr = self.reprocess_ccs(cc_arr, probs_arr)

        return cc_arr, probs_arr


    def adapt_simple_single_case(self, min_prob: float = 0.15):
        """ Simple adaptation - keeping the same form as before and just adapting the probability and possibly
            removing slices from larger connected components.
        Logic for removing slices:
            Look at the cc_id and the slice_num for each row in the results.
            If y_probs is below 0.15, then check if that cc has any slices with y_probs >= 0.15
            If it does, then set the cc_id to 0 for that slice, if not then leave it as is
            We want to remove slices from larger CCs, but not remove whole CCs.
            The 3D connected components are then re-computed, and assigned the max probability across its slices.

            The new CC is saved in out_dir with a corresponding csv with
            the following columns: label, p. label corresponds to the CC id and p is the probability for that CC.
        Args:
            min_prob (float): The minimum probability to keep a lesion on a slice. Default is 0.15.
                              If the entire 3D connected component has a max probability below this, then it is NOT
                              affected. We want to adapt lesion extent, but not to remove whole lesions.
        Returns:
            None
        """
        if '_' not in self.subject_group:
            mask_names = [self.masks_to_use[self.subject_group]]
            thresholds = [self.thresh_to_use[self.subject_group]]
            groups = [self.subject_group]
        else:
            # Handle the cases with 3 seqs (T2+STIR+MP2RAGE)
            groups = self.subject_group.split('_')
            mask_names = [self.masks_to_use[grp] for grp in groups]
            thresholds = [self.thresh_to_use[grp] for grp in groups]

        def blank_prob_with_cc(cc_path) -> Tuple[sitk.Image, np.ndarray, np.ndarray]:
            """ Create a blank connected component image with the same shape as the original CC image. """
            cc_im = load_reorient(cc_path, orientation='LAS')  # Consistent orientation (some processing uses slices)
            cc_arr = sitk_to_numpy(cc_im)
            probs_arr = np.zeros_like(cc_arr, dtype=np.float32)
            return cc_im, cc_arr, probs_arr

        # Create blank arrays to store the outputs (we need this as we may have to combine outputs for two sequences)
        cc_im, _, probs_arr_final = blank_prob_with_cc(self.cc_dir / f'thr-{thresholds[0]}' / mask_names[0])
        cc_arr_final = np.zeros_like(probs_arr_final, dtype=np.int32)

        for mask_fname, thresh, grp in zip(mask_names, thresholds, groups):
            # Determine the mask type (T2, mean, MP2RAGE, STIR, PSIR) from the filename 'preds_mean_cc', 'T2_pred_cc', etc.
            mask_types = [seqname for seqname in ['T2', 'mean', 'MP2RAGE', 'STIR', 'PSIR'] if seqname in mask_fname]
            if len(mask_types) != 1:
                raise ValueError(f'Expected exactly one sequence type in mask_fname: {mask_fname}. Found: {mask_types}')
            mask_type = mask_types[0]
            rf_subset = self.preds_df[(self.preds_df[f'mask'] == mask_type) & (self.preds_df['threshold'] == thresh)]
            if rf_subset.empty:
                msg = f'No prediction rows for {mask_type} with threshold {thresh}. Will save blank image.'
                warn(msg)

            # Load the existing connected components for this sequence and binarisation threshold
            cc_path = self.cc_dir / f'thr-{thresh}' / mask_fname
            _, cc_arr, probs_arr = blank_prob_with_cc(cc_path)

            if len(groups) > 1:
                fov_arr = self.get_fov_ims('MP2RAGE')
                fov_arr = fov_arr > 0
                # Keep CCs entirely within MP2RAGE FOV, and those with >50% within FOV. Remove the rest.
                cc_ids, cc_cnts = np.unique(cc_arr, return_counts=True)
                if len(cc_ids) > 1:
                    def process_partial_ccs(cc_arr, cc_ids, cc_cnts):
                        """ Process the connected components to remove those that are partially outside the FOV. """
                        for cc_id, cc_cnt in zip(cc_ids[1:], cc_cnts[1:]):
                            new_cnt = (cc_arr == cc_id).sum()
                            # If the remaining part of the CC is less than 50%, then remove it entirely
                            if new_cnt < 0.5 * cc_cnt:
                                cc_arr[cc_arr == cc_id] = 0
                        return cc_arr

                    if grp == 'MP2RAGE':
                        cc_arr = cc_arr * fov_arr
                        cc_arr = process_partial_ccs(cc_arr, cc_ids, cc_cnts)
                    elif grp == 'STIR' and 'MP2RAGE' in self.subject_group:
                        # Keep STIR CCs that are entirely outside MP2RAGE FOV, and those with >50% outside MP2RAGE FOV.
                        cc_arr = cc_arr * ~fov_arr
                        cc_arr = process_partial_ccs(cc_arr, cc_ids, cc_cnts)
                        # TODO: for CCs partially in MP2RAGE, the above will still remove the part in MP2RAGE, even
                        #     if we wanted to keep the whole thing...

            cc_arr, probs_arr = self.process_cc_probs_simple(rf_subset, cc_arr, probs_arr, min_prob)

            cc_arr_final = np.where(cc_arr > 0, cc_arr, cc_arr_final)  # Combine the connected components
            probs_arr_final = np.where(probs_arr > 0, probs_arr, probs_arr_final)  # Combine the probabilities

        # Re-process CCs because we may have combined multiple sequences (with different CC IDs)
        cc_arr_final, probs_arr_final = self.reprocess_ccs(cc_arr_final, probs_arr_final)

        # Save the new connected components and probabilities
        self.save_outputs(cc_arr_final, probs_arr_final, cc_im, self.out_dir)


    def adapt_full_single_case(self, seqs: Dict[str, str], thresholds: List[float], min_prob: float = 0.15):
        """ Adapt the connected components for a single case.
            Full adaptation - choose the lesion-slice form with the highest probability
            Gather all the overlapping lesions and re-structure them into a dictionary
            Process image-by-image, slice-by-slice, and process the lesions to keep or remove
            Create a blank image and insert the lesion forms per slice (as binary) (and also probs)
            Re-compute connected components & assign the max probability to each new 3D connected component
        Args:
            seqs (Dict): The sequence names to adapt. Keys correspond to the columns in the preds_df, and the values are
                         the filenames in the cc_dir (cc_dir / f'thr-{thresh}' / seqs['T2'] (-> T2_pred_cc.nii.gz))
            thresholds (List): The thresholds to adapt. Should correspond to the 'threshold' column in the preds_df,
                               and the folder names in the cc_dir (cc_dir / f'thr-{thresh}' / T2_pred_cc.nii.gz)
            min_prob (float): The minimum probability to keep a lesion on a slice. Default is 0.15.
        Returns:
            None
        """
        # Read in all the relevant images
        cc_arrs = []
        im_ids = []
        for seq in seqs:
            for thresh in thresholds:
                if not (self.cc_dir / f'thr-{thresh}' / seqs[seq]).exists():
                    continue
                cc_im = load_reorient(self.cc_dir / f'thr-{thresh}' / seqs[seq], orientation='LAS')
                cc_arrs.append(sitk_to_numpy(cc_im))
                im_ids.append((thresh, seq))

        # Get the probabilities for each connected component
        probs_df = self.preds_df
        probs_df = probs_df.rename(columns={'y_probs': 'predicted_prob'})

        # Get all overlapping lesions
        overlapping_lesions_df = self.get_all_overlapping_lesions(cc_arrs, im_ids, ['threshold', 'mask'])
        overlapping_lesions_dict = self.structure_overlap_dict(overlapping_lesions_df, probs_df, ['threshold', 'mask'])

        # Get the lesions to keep, i.e. in case of overlap, take the lesion-slice form with the highest probability
        lesions_to_keep = self.get_lesions_to_keep(overlapping_lesions_dict)

        # Create the new segmentation mask
        seg, probs = self.create_new_seg_mask(cc_arrs, im_ids, lesions_to_keep)

        # Suppress slices in connected components with low probabilities
        cc_arr = label(seg)
        cc_arr, probs = self.suppress_low_prob_slices(cc_arr, probs, min_prob=min_prob)

        # Re-process the connected components
        new_ccs, new_probs = self.reprocess_ccs(cc_arr, probs)

        # Save the new connected components
        self.save_outputs(new_ccs, new_probs, cc_im, self.out_dir)

    def adapt_hybrid_single_case(self, seqs: Dict[str, str], thresholds: List[float], min_prob: float = 0.15):
        raise NotImplementedError

    def run_adaptation(self, adapt_type: str, seqs: Optional[Dict[str, str]] = None,
                       thresholds: Optional[List[float]] = None, min_prob: float = 0.0):
        """ Run the adaptation for all cases/subjects.
        Args:
            adapt_type (str): The type of adaptation to run. Either 'simple' or 'full'.
                                'simple' - keep the same lesion shapes as the source CCs and just adapt the probability.
                                            If processing slicewise, then slices with prob < min_prob are removed.
                                'full' - choose the overlapping lesion-slice form with the highest probability across
                                         all the possible sequences and thresholds.
                                'hybrid' - take the T2 lesion prediction if it exists, otherwise take any non-overlapping
                                            lesion predictions from the other sequences. (Not implemented for 2D)
            seqs (Dict): The sequence names to adapt. Keys correspond to the columns in the preds_df, and the values are
                            the filenames in the cc_dir (cc_dir / f'thr-{thresh}' / seqs['T2'] (-> T2_pred_cc.nii.gz))
            thresholds (List): The thresholds to adapt. Should correspond to the 'threshold' column in the preds_df,
                                 and the folder names in the cc_dir (cc_dir / f'thr-{thresh}' / T2_pred_cc.nii.gz)
            min_prob (float): The minimum probability to keep a lesion on a slice. Default is 0.15. Only used for 2D.
        Returns:
            None
        """
        if adapt_type not in ['simple', 'full', 'hybrid']:
            raise ValueError('adapt_type must be either "simple" or "full".')

        if adapt_type == 'simple':
            self.adapt_simple_single_case(min_prob)
        elif adapt_type == 'full':
            self.adapt_full_single_case(seqs, thresholds, min_prob)
        elif adapt_type == 'hybrid':
            self.adapt_hybrid_single_case(seqs, thresholds, min_prob)


class Predaptor3D(Predaptor):
    def __init__(self, cc_dir: Path, out_dir: Path, anat_dir: Path, preds_df_path: Optional[Path] = None,
                 preds_df: Optional[pd.DataFrame] = None, id_cols: List[str] = None,
                 masks_to_use: Optional[Dict] = None, thresh_to_use: Optional[Dict] = None
                 ):
        id_cols = id_cols if id_cols is not None else ['case_id', 'cc_id', 'threshold']
        super().__init__(cc_dir, out_dir, anat_dir, preds_df_path=preds_df_path, preds_df=preds_df, id_cols=id_cols,
                         masks_to_use=masks_to_use, thresh_to_use=thresh_to_use)
        self.process_slicewise = False

    def process_cc_probs_simple(self, preds_rows: pd.DataFrame, cc_arr: np.ndarray, probs_arr: np.ndarray,
                                min_prob: Optional[float] = None):
        """ Assign probabilities and process connected component shapes for a single case if necessary.
        Args:
            preds_rows (pd.DataFrame): The rows from the preds_df for this case.
            cc_arr (np.ndarray): The 3D array of connected components.
            probs_arr (np.ndarray): The probabilities for each CC (or all zeros to be filled) (same shape as cc_arr)
            min_prob (float): None. Not used for 3D adaptation.
        Returns:
            None
        """
        for _, row in preds_rows.iterrows():
            cc_id = row['cc_id']
            y_prob = row['y_probs']
            # Set the probability for this slice
            probs_arr[cc_arr == cc_id] = y_prob

        return cc_arr, probs_arr

    @staticmethod
    def get_lesions_to_keep(overlapping_lesions_dict: Dict) -> Dict:
        """ Take the T2 mask if it exists, otherwise take any non-overlapping lesion predictions from the other sequences.
        Args:
            overlapping_lesions_dict (Dict): A dictionary with the CC IDs, probabilities and overlap with other CC IDs.
                E.g. {('0.5', 'T2', 5): {'prob': 0.8, 'overlaps': [('0.5', 'STIR', 2), ('0.001', 'T2', 1)]}, ...}
                where the keys here uniquely identify a connected component (thresh, seq, cc_id)
        Returns:
            lesions_to_keep (Dict): A subset of the input overlapping_lesions_dict, containing only the lesions to keep.
        """
        target_thresholds = {'T2': 0.000001, 'MP2RAGE': 0.000001, 'STIR': 0.5, 'PSIR': 0.1, 'mean': 0.1}
        cc_keys = sorted(overlapping_lesions_dict.keys(), key=lambda x: x[1] == 'T2', reverse=True)
        lesions_to_keep = {}
        for key in cc_keys:
            if len(overlapping_lesions_dict) == 0:
                break
            # Skip if it's not the threshold/seq combinations we're looking for
            if key[1] not in target_thresholds or key[0] != target_thresholds[key[1]]:
                continue
            if key in overlapping_lesions_dict:
                overlaps = overlapping_lesions_dict[key]['overlaps']
                lesions_to_keep[key] = overlapping_lesions_dict.pop(key)
                for overlap_key in overlaps:
                    overlapping_lesions_dict.pop(overlap_key, None)

        return lesions_to_keep

    def adapt_hybrid_single_case(self, seqs: Dict[str, str], thresholds: List[float], min_prob: float = 0.15):
        """ Adapt the connected components for a single case.
            Hybrid adaptation - take the T2 lesion prediction if it exists, otherwise take also any non-overlapping
            lesion predictions from the other sequences.
            For now the threshold to use is just based on the best individual results for those seqs:
            T2: 10^-6, MP2RAGE: 10^-6, STIR: 0.5, PSIR: 0.1
        Args:
            seqs (Dict): The sequence names to adapt. Keys correspond to the columns in the preds_df, and the values are
                         the filenames in the cc_dir (cc_dir / f'thr-{thresh}' / seqs['T2'] (-> T2_pred_cc.nii.gz))
            thresholds (List): The thresholds to adapt. Should correspond to the 'threshold' column in the preds_df,
                               and the folder names in the cc_dir (cc_dir / f'thr-{thresh}' / T2_pred_cc.nii.gz)
            min_prob (float): The minimum probability to keep a lesion on a slice. Default is 0.15.
        Returns:
            None
        """
        # Read in all the relevant images
        cc_arrs = []
        im_ids = []
        target_thresholds = {'T2': 0.000001, 'MP2RAGE': 0.000001, 'STIR': 0.5, 'PSIR': 0.1, 'mean': 0.1}
        for seq in seqs:
            for thresh in thresholds:
                if not (self.cc_dir / f'thr-{thresh}' / seqs[seq]).exists():
                    continue
                # Check that it's one of the seq/thresh combinations that we want:
                if seq not in target_thresholds or thresh != target_thresholds[seq]:
                    continue

                cc_im = load_reorient(self.cc_dir / f'thr-{thresh}' / seqs[seq], orientation='LAS')
                cc_arrs.append(sitk_to_numpy(cc_im))
                im_ids.append((thresh, seq))

        # Get the probabilities for each connected component
        probs_df = self.preds_df
        probs_df = probs_df.rename(columns={'y_probs': 'predicted_prob'})

        # Get all overlapping lesions
        overlapping_lesions_df = self.get_all_overlapping_lesions(cc_arrs, im_ids, ['threshold', 'mask'])
        overlapping_lesions_dict = self.structure_overlap_dict(overlapping_lesions_df, probs_df, ['threshold', 'mask'])

        # Get the lesions to keep, i.e. in case of overlap, take the lesion-slice form with the highest probability
        lesions_to_keep = self.get_lesions_to_keep(overlapping_lesions_dict)

        # Create the new segmentation mask
        seg, probs = self.create_new_seg_mask(cc_arrs, im_ids, lesions_to_keep)

        # Re-process the connected components
        new_ccs, new_probs = self.reprocess_ccs(seg, probs)

        # Save the new connected components
        self.save_outputs(new_ccs, new_probs, cc_im, self.out_dir)


def main(args):
    data_dir = args.data_dir / 'tmp'
    out_dir = args.data_dir

    features_df, ids_df = prepare_features(data_dir / args.features_fname)
    test_fold_df = pd.read_csv(args.test_fold_csv) if args.test_fold_csv else None

    preds = get_preds(features_df, ids_df, args.model_dir, test_fold_df=test_fold_df)
    preds.to_csv(data_dir / 'probs.csv', index=False)

    predadapt3D = Predaptor3D(
        cc_dir=data_dir / 'ccs',
        out_dir=out_dir,
        anat_dir=args.anat_dir,
        preds_df=preds,
        masks_to_use={'MP2RAGE': 'preds_mean_cc.nii.gz',
                      'STIR': 'T2_pred_sc-masked_cc.nii.gz',
                      'PSIR': 'T2_pred_sc-masked_cc.nii.gz'},
        thresh_to_use={'MP2RAGE': 0.0001, 'STIR': 0.00001, 'PSIR': 10**-6},
    )

    predadapt3D.run_adaptation(adapt_type='simple', min_prob=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adapt predicted instances for single case, using probabilities from posthoc model.')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Directory to save the final adapted connected components and probabilities. All other '
                             'data created during the process should already be here within a directory called tmp.')
    parser.add_argument('--anat_dir', type=Path, required=True,
                        help='Directory containing the raw anatomical images for the cases - to be used for re-orienting.')
    parser.add_argument('--features_fname', type=str, required=True,
                        help='Filename of the features table to be used by the posthoc model.')
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Directory containing the posthoc models to be used for adaptation. The models should be '
                             'saved as .joblib files. All of the models in this directory will be loaded and used. '
                             'And the output probability will be the average of all the models.')
    parser.add_argument('--test_fold_csv', type=Path, default=None,
                        help='Path to CSV file which tracks the IDs in the left-out test fold for each model. '
                             'This is not used when deployed, but is only used during development, to ensure we '
                             'do not evaluate the models on samples from their training set. The CSV should have '
                             'columns: case_id, test_fold. e.g., the test_fold is 3 for a sample which was in the '
                             'held-out test set for the model fold-3.joblib')

    args = parser.parse_args()

    main(args)
