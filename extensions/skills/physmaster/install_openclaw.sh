#!/bin/bash
# Install PhysMaster skill for OpenClaw
#
# This script copies the physmaster skill package into OpenClaw's skills
# directory. After installation, agents can load it via:
#   use_skill(name="physmaster", action="get_info")
#
# Usage:
#   bash extensions/skills/physmaster/install_openclaw.sh [skills_dir]
#
# Arguments:
#   skills_dir  Path to OpenClaw's skills directory.
#               Default: ./evomaster/skills (relative to current directory)
#
# Examples:
#   bash extensions/skills/physmaster/install_openclaw.sh
#   bash extensions/skills/physmaster/install_openclaw.sh /path/to/evomaster/skills
#   bash extensions/skills/physmaster/install_openclaw.sh ~/my-project/evomaster/skills

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${1:-./evomaster/skills}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "[ERROR] Skills directory not found: $TARGET_DIR"
    echo ""
    echo "Please provide the path to your OpenClaw skills directory:"
    echo "  bash $0 /path/to/evomaster/skills"
    exit 1
fi

DEST="$TARGET_DIR/physmaster"

if [ -d "$DEST" ]; then
    echo "[INFO] Existing installation found at $DEST, updating..."
    rm -rf "$DEST"
fi

cp -r "$SCRIPT_DIR" "$DEST"
# Remove install scripts from the installed copy (not needed inside OpenClaw)
rm -f "$DEST/install_cc.sh" "$DEST/install_openclaw.sh"

echo "[OK] PhysMaster skill installed for OpenClaw."
echo "     Location: $DEST"
echo ""
echo "Usage in config:"
echo "  agents:"
echo "    my_agent:"
echo "      skills: [\"physmaster\"]"
echo ""
echo "Usage at runtime:"
echo "  use_skill(name=\"physmaster\", action=\"get_info\")"
echo "  use_skill(name=\"physmaster\", action=\"run_script\", path=\"run_physmaster.py --query '...'\")"
