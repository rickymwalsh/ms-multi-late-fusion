
# For every subject in rawdata
# make a directory in inference with the subject name
# copy rawdata/sub-[subj_id]/*[T2|PSIR|STIR|MP2RAGE].nii.gz to inference/sub-[subj_id]/ keeping the orig dir structure
# Also copy case_characteristics.tsv to inference/sub-[subj_id]/
# then zip each sub-[subj_id] directory to sub-[subj_id].zip

# Finally, create a separate json file with the input_archive key pointing to the zip file, name it sub-[subj_id].json, and save it in the inference directory.
#{
#	"input_archive": "/home/rwalsh/Documents/MSMultiSpineChallenge/data/inference/sub-[subj_id].zip",
#}

base_dir=/home/rwalsh/Documents/MSMultiSpineChallenge/data/
cd $base_dir/inference

for subj_dir in $base_dir/rawdata/*; do
  subj_name=$(basename $subj_dir)
  echo "Processing subject: $subj_name"

  # Create output directory for the subject
  out_subj_dir="$subj_name/rawdata/$subj_name"
  mkdir -p $out_subj_dir

  # Copy the T2, PSIR, STIR, MP2RAGE images and case_characteristics.tsv
  cp $subj_dir/*T2.nii.gz $out_subj_dir/
  cp $subj_dir/*PSIR.nii.gz $out_subj_dir/ 2>/dev/null || true
  cp $subj_dir/*STIR.nii.gz $out_subj_dir/ 2>/dev/null || true
  cp $subj_dir/*MP2RAGE.nii.gz $out_subj_dir/ 2>/dev/null || true
  cp $base_dir/case_characteristics.tsv $subj_name/

  # Zip the subject directory
  zip -r "$subj_name.zip" "$subj_name"

  # Create JSON file with input_archive key
  echo "{\"input_archive\": \"$(pwd)/${subj_name}.zip\"}" > "$subj_name.json"
  # remove the unzipped directory
  rm -rf "$subj_name"
done