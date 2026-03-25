#!/usr/bin/env bash
# Render all TurboQuant animations to GIF.
# Usage:
#   ./render.sh          # low quality (fast preview)
#   ./render.sh -qh      # high quality (for README)

set -e

QUALITY=${1:--ql}   # default: low quality
OUT=../assets

mkdir -p "$OUT"

scenes=(RandomRotation PolarQuant TurboQuantPipeline)
files=(rotation polarquant pipeline)

for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    file="${files[$i]}"
    echo "Rendering $scene..."
    manim "$QUALITY" scenes.py "$scene" --format gif --media_dir /tmp/manim_media
    cp /tmp/manim_media/videos/scenes/*/"$scene.gif" "$OUT/$file.gif" 2>/dev/null || \
    find /tmp/manim_media -name "${scene}.gif" -exec cp {} "$OUT/$file.gif" \;
    echo "  → assets/$file.gif"
done

echo ""
echo "Done. Add to README:"
for file in "${files[@]}"; do
    echo "  ![](assets/$file.gif)"
done
