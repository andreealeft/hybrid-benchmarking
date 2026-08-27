# The browser version

The same tool, running inside a browser tab instead of on your machine. It is
for anyone who would rather not install anything, and it needs no terminal, no
Python and no download.

**What it is.** The page is the one the installed version serves, byte for
byte. `boot.js` starts Python inside the tab, installs the ordinary wheel into
it, and replaces `fetch` so that the handful of `/api` addresses the page calls
are answered by `api.py` instead of by a local server. The library is not
modified for the browser and does not know it is in one.

**What changes, and what does not.** Your data still never leaves your machine:
instances, logs and answers are computed in the tab and sent nowhere. The
offline promise is the one thing that weakens, and it is stated on the page: the
first visit downloads Python and its libraries from a public mirror, which the
installed version never does. After that the browser caches them.

**A file from your disk** is copied into Python's own filesystem inside the tab
by the picker beside the file field, and the ordinary reader then reads it from
there. Nothing is uploaded.

## Building it

    sh pages/build.sh

which builds the wheel, copies the page out of the package, injects the one
script tag, and writes the result into `docs/`. That directory is what GitHub
Pages serves, straight from this branch, so publishing needs no workflow and no
build server: run the script, commit what it wrote, push.

`pages/workflow-for-later.yml` is the workflow that would do it instead, running
the test suite first and building on every push. It is parked here rather than
in `.github/workflows` because the token in use cannot create workflow files.
Move it into place when that is no longer true.
