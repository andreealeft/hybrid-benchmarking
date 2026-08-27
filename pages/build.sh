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

cp pages/api.py docs/api.py

# boot.js names the wheel it loads, and that name carries the version, so it is
# written in here from the file that was actually built rather than kept as a
# constant somebody has to remember to edit.  It is written before the page
# that loads it, because the page now has to name its content.
python3 - <<'PY'
import pathlib
wheels = sorted(pathlib.Path("docs").glob("*.whl"))
assert len(wheels) == 1, "expected exactly one wheel, found {}".format(wheels)
boot = pathlib.Path("pages/boot.js").read_text()
assert "@WHEEL@" in boot, "boot.js has no placeholder to fill"
pathlib.Path("docs/boot.js").write_text(boot.replace("@WHEEL@", wheels[0].name))
print("boot.js points at", wheels[0].name)
PY

# GitHub Pages answers every file with `cache-control: max-age=600` and no
# revalidation, and it cannot be told otherwise: a branch build has no headers
# to set.  A browser with the page already open therefore keeps running the old
# boot.js for ten minutes without ever asking whether there is a new one, which
# is exactly how this looked from the outside the first time: pushed, live,
# confirmed by fetching the file, and still the old behaviour on the machine
# that was actually looking at it.  So the page names the script by its
# content, and a changed script becomes a URL no cache has seen.  Same failure
# as pip's cache, answered the same way: put what changed into the name.
python3 - <<'PY'
import hashlib, pathlib
digest = hashlib.sha256(pathlib.Path("docs/boot.js").read_bytes()).hexdigest()[:12]
source = pathlib.Path("src/hybrid_benchmarking/static/index.html").read_text()
marker = "<script>"
assert marker in source
page = source.replace(
    marker, '<script src="boot.js?v={}"></script>\n<script>'.format(digest), 1)
pathlib.Path("docs/index.html").write_text(page)
print("index.html copied, boot.js injected as boot.js?v=" + digest)
PY
touch docs/.nojekyll
echo "built"
