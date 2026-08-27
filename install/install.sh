#!/bin/sh
# Install the tool and put an icon on the Desktop.
#
#   curl -fsSL https://raw.githubusercontent.com/andreealeft/hybrid-benchmarking/main/install/install.sh | sh
#
# Everything lands in one folder of its own, so nothing else on the machine is
# touched and uninstalling is deleting that folder and the icon.  Read this
# file before running it: that is why it is one file and why it is plain.

set -e

SOURCE="https://github.com/andreealeft/hybrid-benchmarking/archive/refs/heads/main.zip"

# Where a machine of this kind keeps such a thing.  One folder either way, so
# uninstalling is deleting it and the icon.
case "$(uname -s)" in
    Darwin)
        HOME_DIR="$HOME/Library/Application Support/hybrid-benchmarking" ;;
    *)
        HOME_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hybrid-benchmarking" ;;
esac
VENV="$HOME_DIR/venv"

echo ""
echo "  Hybrid benchmarking"
echo "  ==================="
echo ""

# ---------------------------------------------------------------- Python
PY=/usr/bin/python3
[ -x "$PY" ] || PY=$(command -v python3 || true)
if [ -z "$PY" ] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo "  This needs Python 3.9 or newer, which is not here yet."
    case "$(uname -s)" in
        Darwin)
            echo "  Opening the download page. Install it, then run this again."
            open "https://www.python.org/downloads/macos/" 2>/dev/null || true ;;
        *)
            echo "  Install it with your package manager, then run this again:"
            echo ""
            echo "      sudo apt install python3 python3-venv     # Debian, Ubuntu, Mint"
            echo "      sudo dnf install python3                  # Fedora"
            echo "      sudo pacman -S python                     # Arch" ;;
    esac
    exit 1
fi

# ---------------------------------------------------------------- the tool
echo "  Installing into a folder of its own. This takes a minute."
mkdir -p "$HOME_DIR"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade --quiet pip
"$VENV/bin/python" -m pip install --upgrade --quiet "$SOURCE"

# ---------------------------------------------------------------- the icon
# Written here rather than downloaded, which is the point: something made on
# this machine carries no quarantine flag, so macOS never questions it.  A
# download would be refused with a message about malware until somebody dug
# through System Settings, and that is the barrier this path exists to avoid.

if [ "$(uname -s)" = "Darwin" ]; then

DESKTOP_APP="$HOME/Desktop/Hybrid benchmarking.app"
rm -rf "$DESKTOP_APP" "$HOME/Desktop/Stop hybrid benchmarking.app"
mkdir -p "$DESKTOP_APP/Contents/MacOS"

cat > "$DESKTOP_APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Hybrid benchmarking</string>
  <key>CFBundleDisplayName</key><string>Hybrid benchmarking</string>
  <key>CFBundleIdentifier</key><string>de.uni-hannover.hybrid-benchmarking</string>
  <key>CFBundleVersion</key><string>3</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>open-it</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cat > "$DESKTOP_APP/Contents/MacOS/open-it" <<'RUN'
#!/bin/sh
# The icon is a doorbell, not the house: it asks the tool to open, and the tool
# starts itself detached if it is not already running.  An icon that stayed
# running would be one the Finder refuses to launch a second time.
LOG="$HOME/Library/Logs/hybrid-benchmarking.log"
VENV="$HOME/Library/Application Support/hybrid-benchmarking/venv"
echo "--- opened $(date)" >> "$LOG"
if ! /usr/bin/curl -s -o /dev/null --max-time 1 http://127.0.0.1:8765/; then
    /usr/bin/osascript -e 'display notification "Starting. Your browser will open in a few seconds." with title "Hybrid benchmarking"' >/dev/null 2>&1
fi
exec "$VENV/bin/hybrid-benchmarking" open >> "$LOG" 2>&1
RUN
chmod +x "$DESKTOP_APP/Contents/MacOS/open-it"

else

# On Linux the same idea is a launcher entry: one in the applications menu, and
# a copy on the Desktop for anyone who looks there first.  Newer desktops want
# it marked as trusted before they will run it, which gio does where it exists.
ENTRY="$HOME/.local/share/applications/hybrid-benchmarking.desktop"
mkdir -p "$(dirname "$ENTRY")"
cat > "$ENTRY" <<ENTRYEOF
[Desktop Entry]
Type=Application
Name=Hybrid benchmarking
Comment=Resource estimates for quantum algorithms
Exec=$VENV/bin/hybrid-benchmarking open
Terminal=false
Categories=Science;Education;
ENTRYEOF
chmod +x "$ENTRY"

DESKTOP_DIR=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
if [ -d "$DESKTOP_DIR" ]; then
    cp "$ENTRY" "$DESKTOP_DIR/hybrid-benchmarking.desktop"
    chmod +x "$DESKTOP_DIR/hybrid-benchmarking.desktop"
    gio set "$DESKTOP_DIR/hybrid-benchmarking.desktop" \
        metadata::trusted true >/dev/null 2>&1 || true
fi

fi

echo ""
echo "  Done. There is now an icon called Hybrid benchmarking: on the Desktop,"
echo "  and in the applications menu if this machine has one."
echo "  Double-click it whenever you want the tool: it opens in your browser."
echo "  It keeps itself up to date, so this is the last time you need a terminal."
echo ""
echo "  Opening it now."
"$VENV/bin/hybrid-benchmarking" open >/dev/null 2>&1 || true
