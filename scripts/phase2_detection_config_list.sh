set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/official.txt}"
RUN_PREFIX="${2:-phase2-official}"
OUTPUT_FILE="${3:-logs/JBShield-D_phase2.log}"
JAILBREAKS="${4:-}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"
: > "${OUTPUT_FILE}"

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="${CONFIG%%#*}"
    CONFIG="${CONFIG//$'\r'/}"
    CONFIG="${CONFIG#"${CONFIG%%[![:space:]]*}"}"
    CONFIG="${CONFIG%"${CONFIG##*[![:space:]]}"}"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(basename "${CONFIG}")"
    NAME="${NAME%.*}"
    RUN_ID="${RUN_PREFIX}-${NAME}"

    echo "===== ${CONFIG} =====" >> "${OUTPUT_FILE}"
    EXTRA_ARGS=()
    if [[ -n "${JAILBREAKS}" ]]; then
        EXTRA_ARGS+=(--jailbreaks "${JAILBREAKS}")
    fi
    if ! uv run python -u detection.py \
        --config "${CONFIG}" \
        --audit-log \
        --phase2 \
        --run-id "${RUN_ID}" \
        "${EXTRA_ARGS[@]}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: Phase2 detection failed. Check GPU memory, model files, gated model access, span mapping, or model-specific runtime support." >> "${OUTPUT_FILE}"
        continue
    fi
done < "${LIST_FILE}"
