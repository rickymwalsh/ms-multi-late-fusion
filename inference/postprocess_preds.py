
import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label
from typing import List
from copy import deepcopy

from im_utils import resample_to_ref, new_image_from_ref, dilate_slicewise, sitk_to_numpy


def check_exists(paths: List[Path], error=True):
    """ Check if multiple paths exist. Can raise an error or simply print a message.
    Args:
        paths - List of paths to check
        error - If True, raise an error if any path does not exist. If False, print a message.
    Returns:
        True if all paths exist, False otherwise.
    """
    not_exists = [p for p in paths if not p.exists()]
    if not_exists:
        msg = f'The following directories do not exist: {not_exists}'
        if error:
            raise FileNotFoundError(msg)
        else:
            print(msg)

    return len(not_exists) == 0


def largest_cc_mask(mask: np.ndarray) -> np.ndarray:
    """Get the largest connected component of a binary mask.
    Args:
        mask: 3D numpy array of the binary mask.
    Returns:
        mask_largest: 3D numpy array of the largest connected component.
    """
    cc, num_cc = label(mask, return_num=True)
    if num_cc > 1:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        largest_cc = np.argmax(sizes)
        mask_largest = (cc == largest_cc).astype(np.uint8)
    else:
        mask_largest = mask

    return mask_largest


def union_sc_seg(sc_seg_arr: np.ndarray) -> np.ndarray:
    """ For each sagittal slice with a spinal cord segmentation, take the union of all the slices.
    Args:
        sc_seg_arr: 3D numpy array of the spinal cord binary segmentation.
    Returns:
        sc_seg_union: 3D numpy array of the union of the spinal cord segmentation.
    """
    sc_seg_union_slice = np.any(sc_seg_arr, axis=0).astype(np.uint8)
    sc_seg_union = np.zeros_like(sc_seg_arr)
    for i in range(sc_seg_union.shape[0]):
        if np.any(sc_seg_arr[i]):
            sc_seg_union[i] = sc_seg_union_slice

    return sc_seg_union


def postprocess_sc_segs(filepath: Path) -> sitk.Image:
    """Process a spinal cord segmentation. 1) Apply 3D closing, 2) take the largest connected component,
    3) apply opening slicewise, 4) take the union of the segmentation across all sagittal slices, and
    5) apply dilation slice-wise (5x5 square).
    Args:
        filepath: Path to the spinal cord segmentation file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sc_seg_im = sitk.ReadImage(filepath)
    # Need to have axes in specific order as we apply operation to each sagittal slice.
    sc_seg_im = sitk.DICOMOrient(sc_seg_im, 'LAS')
    sc_seg_arr = sitk_to_numpy(sc_seg_im)

    # Apply closing
    sc_seg_arr = binary_closing(sc_seg_arr, footprint=np.ones((3, 3, 3)))

    # Take the largest connected component
    largest_cc = largest_cc_mask(sc_seg_arr)

    # Apply opening slicewise to remove small isolated regions on each slice
    sc_seg_arr = np.zeros(sc_seg_arr.shape, dtype=np.uint8)
    for i in range(sc_seg_arr.shape[0]):
        sc_seg_arr[i] = binary_opening(largest_cc[i], footprint=np.ones((3, 3)))

    # Take union of seg across all sagittal slices
    sc_seg_arr = union_sc_seg(sc_seg_arr)

    # Apply dilation slice-wise
    sc_seg_arr = dilate_slicewise(sc_seg_arr, dilation_element=np.ones((5, 5)), slice_axis=0, multi_values=False)

    # Write to file
    sc_seg_im = new_image_from_ref(sc_seg_arr, sc_seg_im)
    sitk.WriteImage(sc_seg_im, str(filepath).replace('.nii.gz', '_processed.nii.gz'))

    return sc_seg_im


def mean_ims(image_list: List[sitk.Image]) -> sitk.Image:
    """
    Compute the mean across several images.
    Args:
        image_list - List containing multiple SITK images to combine by taking the mean across all images.
                     All images should be in the same space, i.e. same geometry etc.
    Returns:
        im_mean - the resulting image after taking the mean of the existing images (sitk.Image)
    """
    im_sum = deepcopy(image_list[0])
    for im in image_list[1:]:
        im_sum += im
    return im_sum / len(image_list)


def main(args):
    preds_dir = args.preds_dir / 'tmp'

    # Get the paths to all the files for this subject
    pred_paths = [preds_dir / pred_name for pred_name in args.preds_names if (preds_dir / pred_name).exists()]
    # Check if the case has T2 + MP2RAGE + STIR -> special treatment compared to two-sequence cases
    case_3seq = any('MP2RAGE' in str(p.name) for p in pred_paths) & any('STIR' in str(p.name) for p in pred_paths)

    # Load the images
    preds = [sitk.ReadImage(p) for p in pred_paths]
    preds = [resample_to_ref(pred, preds[0], interpolator=sitk.sitkLinear, dtype=preds[0].GetPixelID())
             for pred in preds]

    # Process the SC seg
    t2_sc_seg = postprocess_sc_segs(preds_dir / 'T2_sc_seg.nii.gz')
    t2_sc_seg = resample_to_ref(t2_sc_seg, preds[0], interpolator=sitk.sitkNearestNeighbor, dtype=preds[0].GetPixelID())

    # Mask each pred by the t2 SC seg
    preds = [pred * t2_sc_seg for pred in preds]
    for pred, fpath in zip(preds, pred_paths):
        new_path = str(fpath).replace('.nii.gz', '_sc-masked.nii.gz')
        sitk.WriteImage(pred, new_path)

    # If the case has 3 sequences, remove STIR when calculating the mean pmap
    if case_3seq:
        preds = [pred for pred, fpath in zip(preds, pred_paths) if 'STIR' not in str(fpath.name)]

    preds_mean = mean_ims(preds)
    sitk.WriteImage(preds_mean, preds_dir / 'preds_mean.nii.gz')

    # Process the field of view images, either resampling or warping to T2.
    fov_paths_dict = {
        fov_name: {
            'path': preds_dir / fov_name,
            'warp': preds_dir / fov_name.replace('_FOV.nii.gz', '_to_T2_warp.nii.gz')
        }
        for fov_name in args.fov_names if (preds_dir / fov_name).exists()
    }

    for fov_name, fov_paths in fov_paths_dict.items():
        fov_im = sitk.ReadImage(fov_paths['path'])
        if fov_paths['warp'].exists():
            warp_im = sitk.ReadImage(fov_paths['warp'], sitk.sitkVectorFloat64)
            # Create a transform from the warp image
            transform = sitk.DisplacementFieldTransform(warp_im)
        else:
            transform = sitk.Transform()
        # Resample the FOV image to the T2 space
        fov_im_resampled = resample_to_ref(fov_im, preds[0], interpolator=sitk.sitkNearestNeighbor,
                                           dtype=fov_im.GetPixelID(), transform=transform)
        # Write the resampled FOV image to file
        new_fov_path = str(fov_paths['path']).replace('.nii.gz', '_resampled.nii.gz')
        sitk.WriteImage(fov_im_resampled, new_fov_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess predictions for the MSMultiSpineChallenge')
    parser.add_argument('--preds_dir', '-p', type=Path, required=True,
                        help='Path to the directory containing the predictions to postprocess')
    parser.add_argument('--preds_names', '-n', type=str, nargs='+', help='Names of the prediction files to postprocess',
                        required=True)

    args = parser.parse_args()
    # The field of view ims will also be processed here with SITK, because SCT resampling does not allow a default value
    #  when out of bounds.
    args.fov_names = ['MP2RAGE_FOV.nii.gz', 'STIR_FOV.nii.gz', 'PSIR_FOV.nii.gz']

    main(args)
