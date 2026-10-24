#!/usr/bin/env bash
# Fetch the Unitree G1 model (with hands) from MuJoCo Menagerie into assets/unitree_g1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/assets/unitree_g1"
if [ -f "$DEST/scene_with_hands.xml" ]; then
  echo "G1 assets already present at $DEST"; exit 0
fi
TMP="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
git -C "$TMP/menagerie" sparse-checkout set unitree_g1
mkdir -p "$ROOT/assets"
cp -r "$TMP/menagerie/unitree_g1" "$DEST"
rm -rf "$TMP"
echo "G1 assets installed at $DEST"
