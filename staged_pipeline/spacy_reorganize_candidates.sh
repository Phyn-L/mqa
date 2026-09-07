conda run -n shine python spacy_reorganize_candidates.py \
  --dataset squad \
  --batch-size 64 \
  --sortish-window-size 2048 \
  --sortish-seed 42 \
  --token-budget 65536 \
  --min-coverage 1.0
