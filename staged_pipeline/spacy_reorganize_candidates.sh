conda run -n shine python spacy_reorganize_candidates.py \
  --dataset squad \
  --batch-size 8 \
  --max-new-tokens 6144 \
  --sortish-window-size 1024 \
  --sortish-seed 42 \
  --token-budget 49152
