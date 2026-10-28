#!/bin/bash
# StructSense AI Launcher (macOS) - Double-click in Finder to start the application.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/structsense/run.command"
