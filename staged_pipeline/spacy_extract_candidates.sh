
conda run -n shine python spacy_extract_candidates.py \
  --dataset coqa \
  --spacy-model en_core_web_sm \
  --batch-size 256 \
  --processes 8
