#!/bin/bash

# Args:
#   1- in_dir: path to a single case
#   2- out_dir: path to the output directory for a single case
#   3- task: either lesion or sc  (lesion_ms or seg_sc_contrast_agnostic)

# Process input args
in_path=${1}
out_path=${2}

tmp_path=${out_path}/tmp
mkdir -p $tmp_path

if [[ ${3} =~ "lesion" ]]; then
  task="lesion_ms"
  thr_arg="-thr 0"
  tta_arg="-test-time-aug"
  suffix="_pred"
elif [[ ${3} =~ "sc" ]]; then
  task="spinalcord"
  thr_arg=""
  tta_arg=""
  suffix="_sc_seg"
else
  echo "Task argument (arg 3) must be either lesion_ms or seg_sc_contrast_agnostic. Exiting."
  exit 0
fi

# Process the T2 pred first
for t2_im in $in_path/*T2.nii.gz; do
  # Get pred
  CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 sct_deepseg $task \
    -i $t2_im \
    -o $out_path/tmp/T2${suffix}.nii.gz \
    $thr_arg \
    $tta_arg
done
# Then process the other modalities
for ftype in STIR PSIR MP2RAGE; do
  if [ $ftype == "MP2RAGE" ];then
    # Do not need TTA for MP2RAGE - it does not improve perf
    tta_arg=""
  fi

  fpath=$in_path/*${ftype}.nii.gz
  # Get pred if the file exists
  if [ ! -f $fpath ]; then
    echo "File $fpath does not exist. Skipping modality $ftype."
    continue
  fi
  CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 sct_deepseg $task \
    -i $fpath \
    -o $out_path/tmp/${ftype}${suffix}_origspace.nii.gz \
    $thr_arg \
    $tta_arg

  if ! [ -f $out_path/${ftype}${suffix}_resampled.nii.gz ]; then
    if [[ ${3} =~ "lesion" ]]; then
      interp="linear"  # If we don't initially threshold the segmentation, then we use linear interpolation
    else
      interp="nn"  # use nearest neighbour interpolation if the segmentation is already thresholded
    fi
    sct_resample  -i $out_path/tmp/${ftype}${suffix}_origspace.nii.gz \
                  -o $out_path/tmp/${ftype}${suffix}_resampled.nii.gz \
                  -ref $t2_im \
                  -x $interp
  fi
done
