#!/opt/homebrew/bin/bash
# build.sh — Compile Java agents for FoW Chess
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LUDII_JAR="$PROJECT_DIR/ludii/Ludii-1.3.14.jar"
SRC_DIR="$PROJECT_DIR/ludii/agents/src"
COMPILED_DIR="$PROJECT_DIR/ludii/agents/compiled"
JAR_DIR="$PROJECT_DIR/ludii/agents/jars"
OUTPUT_JAR="$JAR_DIR/agents.jar"
CLEAN=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-clean) CLEAN=false; shift ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " Building FoW Chess AI Agents"
echo " Source: $SRC_DIR"
echo " Output: $OUTPUT_JAR"
echo " Clean:  $CLEAN"
echo "============================================================"

[[ ! -f "$LUDII_JAR" ]] && { echo "ERROR: Ludii JAR not found at $LUDII_JAR"; exit 1; }
command -v javac &>/dev/null || { echo "ERROR: javac not in PATH. Install JDK 11+."; exit 1; }

JAVA_VERSION=$(javac -version 2>&1 | awk '{print $2}' | cut -d. -f1)
echo "Java version: $JAVA_VERSION"

mkdir -p "$JAR_DIR"

if [[ "$CLEAN" == "true" ]]; then
    echo "[1/4] Cleaning compiled directory..."
    rm -rf "$COMPILED_DIR"
fi
mkdir -p "$COMPILED_DIR"

echo "[2/4] Collecting source files..."
SOURCES=()
while IFS= read -r f; do
    SOURCES+=("$f")
done < <(find "$SRC_DIR" -name "*.java")
echo "      Found ${#SOURCES[@]} source file(s):"
for s in "${SOURCES[@]}"; do echo "        ${s##*/}"; done

echo "[3/4] Compiling..."
javac \
    -cp "$LUDII_JAR" \
    -d  "$COMPILED_DIR" \
    "${SOURCES[@]}" 2>&1

echo "[4/4] Packaging JAR..."
jar cf "$OUTPUT_JAR" -C "$COMPILED_DIR" .

echo ""
echo "============================================================"
echo " Build successful: $OUTPUT_JAR"
echo " Classes: $(jar tf "$OUTPUT_JAR" | grep -c '\.class$')"
echo "============================================================"
