#!/bin/bash


in_path=${1}
out_path=${2}
tmp_path=${out_path}/tmp


t2_path=$in_path/*T2.nii.gz
if [ $(ls $in_path/*STIR.nii.gz | wc -l) -eq 1 ]; then
  moving_path=$in_path/*STIR.nii.gz
  ftype="STIR"
elif [ $(ls $in_path/*PSIR.nii.gz | wc -l) -eq 1 ]; then
  moving_path=$in_path/*PSIR.nii.gz
  ftype="PSIR"
# Don't apply registration for MP2RAGE
else
  echo "No moving image found for $sub_name. Exiting."
  exit 1
fi

# Crop the images around the spinal cord to make registration easier and more accurate on SC region
t2_path_crop=$out_path/tmp/T2_crop.nii.gz
moving_path_crop=$out_path/tmp/${ftype}_crop.nii.gz

sct_create_mask -i $t2_path -p centerline,$out_path/tmp/T2_sc_seg.nii.gz -size 35mm \
                -o $out_path/tmp/T2_mask.nii.gz
sct_create_mask -i $moving_path -p centerline,$out_path/tmp/${ftype}_sc_seg_origspace.nii.gz -size 35mm \
                -o $out_path/tmp/${ftype}_mask.nii.gz

sct_crop_image -i $t2_path -m $out_path/tmp/T2_mask.nii.gz -o $t2_path_crop
sct_crop_image -i $moving_path -m $out_path/tmp/${ftype}_mask.nii.gz -o $moving_path_crop

# Register using the DL method of the SCT
sct_register_multimodal -i $moving_path_crop -d $t2_path_crop -o $out_path/tmp/${ftype}_to_T2.nii.gz \
                      -param step=1,type=im,algo=dl -dseg $out_path/tmp/T2_sc_seg.nii.gz \
                      -owarp $out_path/tmp/${ftype}_to_T2_warp.nii.gz \
                      -owarpinv $out_path/tmp/T2_to_${ftype}_warp.nii.gz

# Apply transform to pred -> T2
sct_apply_transfo -i $out_path/tmp/${ftype}_pred_origspace.nii.gz \
                  -d $t2_path -w $out_path/tmp/${ftype}_to-T2-warp.nii.gz \
                  -o $out_path/tmp/${ftype}_pred_to-T2-warped.nii.gz -x linear


