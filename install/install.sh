#!/bin/sh
# What the installer app runs.  No terminal, so everything it has to say it
# says in a dialog, and everything it does is written to a log.
#
# It installs into an environment of its own under Application Support rather
# than into the system Python, so nothing else on the machine is touched and
# removing the tool is removing one folder.

LOG="$HOME/Library/Logs/hybrid-benchmarking-install.log"
HOME_DIR="$HOME/Library/Application Support/hybrid-benchmarking"
VENV="$HOME_DIR/venv"
SOURCE="https://github.com/andreealeft/hybrid-benchmarking/archive/refs/heads/main.zip"

say() { /usr/bin/osascript -e "display notification \"$1\" with title \"Hybrid benchmarking\"" >/dev/null 2>&1; }
tell() { /usr/bin/osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"Hybrid benchmarking\"" >/dev/null 2>&1; }

echo "--- install $(date)" >> "$LOG"

# ---------------------------------------------------------------- Python
PY=/usr/bin/python3
[ -x "$PY" ] || PY=$(command -v python3)
if [ -z "$PY" ] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >>"$LOG" 2>&1; then
    tell "This needs Python, which is not on this Mac yet.\n\nThe download page is opening now. Install Python, then open this installer again."
    open "https://www.python.org/downloads/macos/"
    exit 0
fi

# ---------------------------------------------------------------- the tool
say "Installing. This takes a minute."
mkdir -p "$HOME_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV" >>"$LOG" 2>&1 || {
        tell "The environment could not be created. The details are in Console, under hybrid-benchmarking-install."
        exit 1
    }
fi
"$VENV/bin/python" -m pip install --upgrade --quiet pip >>"$LOG" 2>&1
"$VENV/bin/python" -m pip install --upgrade --quiet "$SOURCE" >>"$LOG" 2>&1 || {
    tell "The install did not finish. The details are in Console, under hybrid-benchmarking-install."
    exit 1
}

# ---------------------------------------------------------------- the icons
make_app() {
    NAME="$1"; RUNNER="$2"; BODY="$3"
    APP="$HOME/Desktop/$NAME.app"
    rm -rf "$APP"
    mkdir -p "$APP/Contents/MacOS"
    cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$NAME</string>
  <key>CFBundleDisplayName</key><string>$NAME</string>
  <key>CFBundleIdentifier</key><string>de.uni-hannover.hybrid-benchmarking.$RUNNER</string>
  <key>CFBundleVersion</key><string>2</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>$RUNNER</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST
    printf '%s\n' "$BODY" > "$APP/Contents/MacOS/$RUNNER"
    chmod +x "$APP/Contents/MacOS/$RUNNER"
}

# The icon is a doorbell, not the house: it asks the tool to open, and the tool
# starts itself detached if it is not already running.  An icon that stayed
# running would be one the Finder refuses to launch a second time.
make_app "Hybrid benchmarking" "open-it" '#!/bin/sh
LOG="$HOME/Library/Logs/hybrid-benchmarking.log"
VENV="$HOME/Library/Application Support/hybrid-benchmarking/venv"
echo "--- opened $(date)" >> "$LOG"
if ! /usr/bin/curl -s -o /dev/null --max-time 1 http://127.0.0.1:8765/; then
    /usr/bin/osascript -e '"'"'display notification "Starting. Your browser will open in a few seconds." with title "Hybrid benchmarking"'"'"' >/dev/null 2>&1
fi
exec "$VENV/bin/hybrid-benchmarking" open >> "$LOG" 2>&1'

make_app "Stop hybrid benchmarking" "stop-it" '#!/bin/sh
PIDS=$(/usr/sbin/lsof -ti :8765 -sTCP:LISTEN 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill 2>/dev/null
    /usr/bin/osascript -e '"'"'display notification "Stopped." with title "Hybrid benchmarking"'"'"' >/dev/null 2>&1
else
    /usr/bin/osascript -e '"'"'display notification "It was not running." with title "Hybrid benchmarking"'"'"' >/dev/null 2>&1
fi'

tell "Installed.\n\nThere is now an icon on your Desktop called Hybrid benchmarking. Double-click it whenever you want the tool: it opens in your browser.\n\nIt keeps itself up to date, and the icon beside it stops it."
open "$HOME/Desktop"
