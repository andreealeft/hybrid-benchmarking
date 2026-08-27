#!/bin/sh
# Rebuild the browser build.  Run from the repository root on the pages
# branch: sh pages/build.sh
#
# Sources live in pages/ and the built site lands in docs/, which is one of the
# two places GitHub Pages will serve from a branch directly.  That avoids a
# workflow, and therefore avoids needing workflow scope on a token.
#
# The page itself is copied from the package rather than edited here, so the
# browser build cannot drift from the one the local server hands out.  The only
# change made to it is the line that loads boot.js, which is what replaces the
# server with Python running in the tab.
set -e

mkdir -p docs
rm -f docs/*.whl
python3 -m pip wheel . --no-deps -w docs -q

python3 - <<'PY'
import pathlib
source = pathlib.Path("src/hybrid_benchmarking/static/index.html").read_text()
marker = "<script>"
assert marker in source
page = source.replace(marker, '<script src="boot.js"></script>\n<script>', 1)
pathlib.Path("docs/index.html").write_text(page)
print("index.html copied, boot.js injected")
PY

cp pages/api.py docs/api.py
cp pages/boot.js docs/boot.js
touch docs/.nojekyll
echo "built"
