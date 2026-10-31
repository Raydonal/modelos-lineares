#!/usr/bin/env bash
# Envia site/_site para a branch gh-pages. Rode `make site` antes.
set -euo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
URL=https://github.com/Raydonal/modelos-lineares.git
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
if git clone -q --depth 1 --branch gh-pages "$URL" "$TMP" 2>/dev/null; then
  find "$TMP" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
else
  git -C "$TMP" init -q -b gh-pages
  git -C "$TMP" remote add origin "$URL"
fi
cp -r "$RAIZ"/site/_site/. "$TMP"/
touch "$TMP"/.nojekyll
git -C "$TMP" add -A
git -C "$TMP" commit -q -m "Site $(git -C "$RAIZ" rev-parse --short HEAD)"
git -C "$TMP" push -q origin gh-pages
