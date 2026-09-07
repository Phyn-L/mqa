conda run -n shine python spacy_reorganize_candidates.py \
  --dataset coqa \
  --batch-size 256 \
  --sortish-window-size 2048 \
  --sortish-seed 42 \
  --token-budget 131072 \
  --min-coverage 1.0
