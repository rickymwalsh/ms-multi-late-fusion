for json_file in /home/rwalsh/Documents/MSMultiSpineChallenge/data/inference/*.json; do
  sudo /home/rwalsh/miniconda3/bin/bosh exec launch descriptor.json json_file --no-pull;
done