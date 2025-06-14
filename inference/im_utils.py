
import SimpleITK as sitk
import numpy as np
from skimage import morphology, filters
from typing import Optional


def sitk_to_numpy(sitk_im: sitk.Image) -> np.ndarray:
    """ Convert SimpleITK image to numpy array. SimpleITK and numpy use different axis orders, so we swap them."""
    return np.swapaxes(sitk.GetArrayFromImage(sitk_im), 0, 2)


def new_image_from_ref(new_data: np.ndarray, ref_im: sitk.Image, numpy_to_sitk=True) -> sitk.Image:
    """ Use new data array but copy spacing, origin and affine from ref_im, with option to swap axes to convert from
    numpy to SimpleITK axis order.
    Args:
        new_data: numpy array containing image data
        ref_im: reference image from which to copy spacing, origin and affine
        numpy_to_sitk: if True, swap axes to convert from numpy to SimpleITK axis order
    Returns:
        new_im: SimpleITK image with new data and copied properties from ref_im
    """
    # SITK uses different axis order to numpy, so we need to swap axes
    if numpy_to_sitk:
        new_data = np.swapaxes(new_data, 0, 2)
    new_im = sitk.GetImageFromArray(new_data)
    new_im.SetSpacing(ref_im.GetSpacing())
    new_im.SetOrigin(ref_im.GetOrigin())
    new_im.SetDirection(ref_im.GetDirection())
    return new_im


def resample_to_ref(im: sitk.Image, ref_im: sitk.Image, transform=sitk.AffineTransform(3), interpolator=sitk.sitkLinear,
                    dtype: Optional[int] = None) -> sitk.Image:
    if dtype is None:
        dtype = im.GetPixelID()
    return sitk.Resample(im, size=ref_im.GetSize(), transform=transform, interpolator=interpolator,
                         outputOrigin=ref_im.GetOrigin(), outputSpacing=ref_im.GetSpacing(),
                         outputDirection=ref_im.GetDirection(), defaultPixelValue=0, outputPixelType=dtype)


def get_bbox_bounds(im: sitk.Image) -> list:
    """ Get the min and max indices of the bounding box of non-zero voxels in im.
    Args:
        im: SimpleITK image.
    Returns:
        bounds: List of tuples containing the min and max indices for each axis,
                [(min_x, max_x), (min_y, max_y), (min_z, max_z)].
    """
    ax0 = np.any(im, axis=(1, 2))
    ax1 = np.any(im, axis=(0, 2))
    ax2 = np.any(im, axis=(0, 1))
    ax0_min, ax0_max = np.where(ax0)[0][[0, -1]]
    ax1_min, ax1_max = np.where(ax1)[0][[0, -1]]
    ax2_min, ax2_max = np.where(ax2)[0][[0, -1]]
    bounds = [(ax0_min, ax0_max), (ax1_min, ax1_max), (ax2_min, ax2_max)]
    # Convert to int (rather than numpy.int64) to avoid errors when saving to json
    bounds = [(int(min_), int(max_)) for min_, max_ in bounds]
    return bounds


def dilate_slicewise(mask_arr: np.ndarray, dilation_element=morphology.disk(1), slice_axis=0, multi_values=False):
    """ Dilates the blobs in a 3D mask array slicewise along the specified axis.
    Args:
        mask_arr: 3D numpy array of the mask.
        dilation_element: Element to use for dilation.
        slice_axis: Axis along which to dilate the blobs.
        multi_values: If True, the mask is allowed to have several values, which are kept when dilated.
    Returns:
        Array with same shape as mask_arr with the blobs dilated along the specified axis.
    """
    if not multi_values and len(np.unique(mask_arr)) > 2:
        raise ValueError('The mask has more than 2 unique values. If multi_values is False, the mask should be binary.')

    func = morphology.binary_dilation if not multi_values else filters.rank.maximum

    dilated_mask = np.zeros_like(mask_arr)
    for i in range(mask_arr.shape[slice_axis]):
        if slice_axis == 0:
            dilated_mask[i] = func(mask_arr[i], footprint=dilation_element)
        elif slice_axis == 1:
            dilated_mask[:, i] = func(mask_arr[:, i], footprint=dilation_element)
        elif slice_axis == 2:
            dilated_mask[:, :, i] = func(mask_arr[:, :, i], footprint=dilation_element)
    return dilated_mask
