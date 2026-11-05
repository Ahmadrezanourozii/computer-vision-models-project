#!/bin/bash
# Upload (or version) the private Kaggle dataset with Original/, Mooney/, metadata.csv and code/vision.
#   kaggle/upload_dataset.sh <token_file> <owner> [message]
set -e
TOK=$(cat "$1"); OWNER=$2; MSG=${3:-update}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
mkdir -p "$STAGE/code/vision"
cp -r "$ROOT/data/Original" "$ROOT/data/Mooney" "$STAGE/"
cp "$ROOT/data/metadata.csv" "$STAGE/"
cp "$ROOT"/vision/*.py "$STAGE/code/vision/"
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "mooney-faces-data", "id": "$OWNER/mooney-faces-data", "licenses": [{"name": "CC0-1.0"}]}
EOF
if KAGGLE_API_TOKEN=$TOK kaggle datasets status "$OWNER/mooney-faces-data" >/dev/null 2>&1; then
  KAGGLE_API_TOKEN=$TOK kaggle datasets version -p "$STAGE" -m "$MSG" --dir-mode zip
else
  KAGGLE_API_TOKEN=$TOK kaggle datasets create -p "$STAGE" --dir-mode zip
fi
rm -rf "$STAGE"
