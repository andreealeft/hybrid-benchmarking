#!/bin/sh
# Double-click this once.  It installs the tool and puts an icon on the Desktop
# that opens it.  Nothing has to be typed.
#
# It is deliberately readable: this is a file somebody downloaded from the
# internet and is about to run, and they should be able to see what it does.

set -e
cd "$(dirname "$0")"

echo ""
echo "  Hybrid benchmarking"
echo "  ==================="
echo ""

# ---------------------------------------------------------------- Python
if command -v python3 >/dev/null 2>&1 && python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    PY=python3
else
    echo "  This needs Python, which is not on this Mac yet."
    echo "  Opening the download page now."
    echo ""
    echo "  Install it, then double-click this file again."
    open "https://www.python.org/downloads/macos/"
    echo ""
    echo "  Press return to close this window."
    read _ignored
    exit 0
fi

# ---------------------------------------------------------------- the tool
echo "  Installing. This takes a minute or two the first time."
echo ""
# From a zip of the repository rather than through git, which most people do
# not have and should not have to install to get a number.
$PY -m pip install --user --upgrade --quiet \
    "https://github.com/andreealeft/hybrid-benchmarking/archive/refs/heads/main.zip" \
    || {
        echo ""
        echo "  The install did not finish. The message above says why."
        echo "  Press return to close this window."
        read _ignored
        exit 1
    }

# ---------------------------------------------------------------- the icon
APP="$HOME/Desktop/Hybrid benchmarking.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Hybrid benchmarking</string>
  <key>CFBundleDisplayName</key><string>Hybrid benchmarking</string>
  <key>CFBundleIdentifier</key><string>de.uni-hannover.hybrid-benchmarking</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>open-it</string>
  <!-- An agent rather than a windowed app.  It has no windows of its own: the
       interface is the browser.  Without this the Dock shows an icon for a
       process that may exit a second later, having only opened a page, and
       macOS then reports that the application is not open anymore, which reads
       as a crash. -->
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/open-it" <<'RUN'
#!/bin/sh
# Open the tool, starting it first if it is not already running.
#
# The work is in "hybrid-benchmarking open", which checks whether it is already
# running, starts it detached if not, and shows it either way.  This icon only
# rings the doorbell: an app that stayed running would be one the Finder
# refuses to launch a second time, since it would merely try to bring it to the
# front and an app with no windows has no front to come to.
#
# An app launched from the Finder gets a bare PATH, so everything is found by
# full path, and whatever goes wrong is written down where it can be read.
LOG="$HOME/Library/Logs/hybrid-benchmarking.log"
URL="http://127.0.0.1:8765/"

PY=/usr/bin/python3
[ -x "$PY" ] || PY=$(command -v python3)
VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
INSTALLED="$HOME/Library/Python/$VERSION/bin/hybrid-benchmarking"

echo "--- opened $(date)" >> "$LOG"

if ! /usr/bin/curl -s -o /dev/null --max-time 1 "$URL"; then
    /usr/bin/osascript -e 'display notification "Starting. Your browser will open in a few seconds." with title "Hybrid benchmarking"' >/dev/null 2>&1
fi

if [ -x "$INSTALLED" ]; then
    exec "$INSTALLED" open >> "$LOG" 2>&1
fi
exec "$PY" -m hybrid_benchmarking.cli open >> "$LOG" 2>&1
RUN
chmod +x "$APP/Contents/MacOS/open-it"

STOP="$HOME/Desktop/Stop hybrid benchmarking.app"
rm -rf "$STOP"
mkdir -p "$STOP/Contents/MacOS"
sed 's/Hybrid benchmarking/Stop hybrid benchmarking/; s/open-it/stop-it/' \
    "$APP/Contents/Info.plist" > "$STOP/Contents/Info.plist"

cat > "$STOP/Contents/MacOS/stop-it" <<'RUN'
#!/bin/sh
# Stop the tool.  It is a local server and nothing else, so this ends the one
# process that is listening and says so.
PIDS=$(/usr/sbin/lsof -ti :8765 -sTCP:LISTEN 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill 2>/dev/null
    /usr/bin/osascript -e 'display notification "Stopped." with title "Hybrid benchmarking"' >/dev/null 2>&1
else
    /usr/bin/osascript -e 'display notification "It was not running." with title "Hybrid benchmarking"' >/dev/null 2>&1
fi
RUN
chmod +x "$STOP/Contents/MacOS/stop-it"

echo ""
echo "  Done."
echo ""
echo "  There is now an icon on your Desktop called Hybrid benchmarking."
echo "  Double-click it whenever you want the tool: it opens in your browser."
echo "  It keeps running quietly after you close the tab, so opening it again"
echo "  is instant. There is a second icon beside it to stop it, and it stops"
echo "  by itself when you restart the Mac."
echo ""
echo "  Press return to close this window."
read _ignored
