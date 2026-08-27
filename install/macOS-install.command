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
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/open-it" <<'RUN'
#!/bin/sh
# Start the tool and open it in the browser.  Quitting this icon stops it.
#
# An app launched from the Finder gets a bare PATH, so the interpreter and the
# command are found by their full paths rather than by name, and anything that
# goes wrong is written down where it can be read afterwards.
LOG="$HOME/Library/Logs/hybrid-benchmarking.log"
PY=/usr/bin/python3
[ -x "$PY" ] || PY=$(command -v python3)

VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
INSTALLED="$HOME/Library/Python/$VERSION/bin/hybrid-benchmarking"

URL=http://127.0.0.1:8765/
echo "--- started $(date)" >> "$LOG"

# Already running from an earlier double-click: show it rather than starting a
# second one, which would fail on the port and look like a broken app.
if /usr/bin/curl -s -o /dev/null --max-time 1 "$URL"; then
    open "$URL"
    exit 0
fi

# It takes a few seconds to start, and a Dock icon with nothing happening is
# how somebody ends up double-clicking three times.
/usr/bin/osascript -e 'display notification "Starting. Your browser will open in a few seconds." with title "Hybrid benchmarking"' >/dev/null 2>&1

if [ -x "$INSTALLED" ]; then
    exec "$INSTALLED" >> "$LOG" 2>&1
fi
exec "$PY" -m hybrid_benchmarking.cli serve >> "$LOG" 2>&1
RUN
chmod +x "$APP/Contents/MacOS/open-it"

echo ""
echo "  Done."
echo ""
echo "  There is now an icon on your Desktop called Hybrid benchmarking."
echo "  Double-click it whenever you want the tool: it opens in your browser."
echo "  To stop it, quit that icon from the Dock."
echo ""
echo "  Press return to close this window."
read _ignored
