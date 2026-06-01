#!/usr/bin/env bash
# Export Reveal.js slides to PDF using decktape.
# Each fragment step is exported as a separate page.
# Interactive Plotly slides are captured in their initial fully-revealed state.
#
# Usage: ./export_pdf.sh [lecture0|lecture1|all]

set -e

SLIDES_DIR="$(cd "$(dirname "$0")/docs" && pwd)"
PORT=8765
TARGET="${1:-all}"

# Start a local HTTP server
python3 -m http.server $PORT --directory "$SLIDES_DIR" &>/dev/null &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
sleep 1  # let server start

export_slide() {
    local file="$1"
    local out="$2"
    local pause="${3:-1500}"
    echo "Exporting $file → $out ..."
    npx decktape reveal \
        --size 1600x900 \
        --pause "$pause" \
        --load-pause 2000 \
        "http://localhost:$PORT/$file" \
        "$SLIDES_DIR/$out"
    echo "  Done: $out"
}

case "$TARGET" in
    lecture0)
        export_slide lecture0.html lecture0.pdf 800
        ;;
    lecture1)
        export_slide lecture1.html lecture1.pdf 1500
        echo "Capturing interactive step slides..."
        node "$SLIDES_DIR/export_steps.js"
        ;;
    lecture3)
        export_slide lecture3.html lecture3.pdf 1500
        ;;
    lecture4)
        export_slide lecture4.html lecture4.pdf 1500
        ;;
    lecture4b)
        export_slide lecture4b.html lecture4b.pdf 1500
        ;;
    lecture5)
        export_slide lecture5.html lecture5.pdf 1500
        ;;
    lecture6)
        export_slide lecture6.html lecture6.pdf 1500
        ;;
    lecture7)
        export_slide lecture7.html lecture7.pdf 2500
        ;;
    all)
        export_slide lecture0.html lecture0.pdf 800
        export_slide lecture1.html lecture1.pdf 1500
        export_slide lecture3.html lecture3.pdf 1500
        export_slide lecture4.html lecture4.pdf 1500
        export_slide lecture4b.html lecture4b.pdf 1500
        export_slide lecture5.html lecture5.pdf 1500
        export_slide lecture6.html lecture6.pdf 1500
        export_slide lecture7.html lecture7.pdf 2500
        echo "Capturing interactive step slides..."
        node "$SLIDES_DIR/export_steps.js"
        ;;
    *)
        echo "Usage: $0 [lecture0|lecture1|lecture3|lecture4|lecture4b|lecture5|lecture6|lecture7|all]"
        exit 1
        ;;
esac

echo "All done."
