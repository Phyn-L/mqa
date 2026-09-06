#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/raw/peerqa/source"
mkdir -p "$OUT"
URL='https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/peerqa-data-v1.0.zip?sequence=5&isAllowed=y'
ARCHIVE="$OUT/peerqa-data-v1.0.zip"
curl -L "$URL" -o "$ARCHIVE"
if file "$ARCHIVE" | grep -qi html; then
  rm -f "$ARCHIVE"
  echo 'PeerQA source returned an HTML anti-bot page; no data was saved.' >&2
  exit 2
fi
unzip -o "$ARCHIVE" -d "$OUT"
rm -f "$ARCHIVE"
echo "PeerQA source extracted to $OUT"
