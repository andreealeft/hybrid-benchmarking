#!/bin/sh
# Build the double-clickable installer for macOS.
#
#     sh install/build-macos-installer.sh
#
# It writes "install/Install hybrid benchmarking.app" and zips it, because a
# .app is a directory and a directory cannot be downloaded from a web page.
# The zip is what people click; macOS unpacks it on download and they open the
# app inside.
set -e
cd "$(dirname "$0")/.."

APP="install/Install hybrid benchmarking.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Install hybrid benchmarking</string>
  <key>CFBundleDisplayName</key><string>Install hybrid benchmarking</string>
  <key>CFBundleIdentifier</key><string>de.uni-hannover.hybrid-benchmarking.installer</string>
  <key>CFBundleVersion</key><string>2</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>install</string>
  <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

cp install/install.sh "$APP/Contents/MacOS/install"
chmod +x "$APP/Contents/MacOS/install"

rm -f install/Install-hybrid-benchmarking-mac.zip
( cd install && zip -qr Install-hybrid-benchmarking-mac.zip "Install hybrid benchmarking.app" )
echo "built install/Install-hybrid-benchmarking-mac.zip"
