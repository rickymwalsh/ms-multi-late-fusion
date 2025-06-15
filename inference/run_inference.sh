#!/bin/bash

# Args:
#   1- in_dir: path to data for a single case
#   2- out_dir: path to the output directory for a single case

# Get the sub directory in raw_data - there should only be one!
in_dir=$(ls -d ${1}/rawdata/sub-*)
out_dir=${2}

# Get the lesion & SC segmentations
bash sct-preds.sh $in_dir $out_dir lesion
bash sct-preds.sh $in_dir $out_dir sc
# Register the other seq to T2
bash register.sh $in_dir $out_dir
# Process the lesion segmentations (masking by the SC after processing the SC masks & compute mean of T2 and other seq)
echo "Processing SC & lesion segmentations ----------------"
python3 postprocess_preds.py --preds_dir $out_dir  \
  --preds_names T2_pred.nii.gz MP2RAGE_pred_resampled.nii.gz PSIR_pred_warped.nii.gz STIR_pred_warped.nii.gz
# Compute the features used in the posthoc model
echo "Computing features for posthoc model ----------------"
python3 feature_engineering.py --pred_dir $out_dir \
  --anat_dir $in_dir \
  --metadata_path $(dirname $(dirname $in_dir))/case_characteristics.tsv \
  --thresholds 0.000001 0.00001 0.0001
# Adapt the instance probabilities using the posthoc model
echo "Adapting the instance probabilities using the posthoc model ----------------"
python3 adapt_preds.py --data_dir $out_dir \
  --anat_dir $in_dir \
  --features_fname features.csv \
  --model_dir XGBClassifier-submission-aug-3D-v1


