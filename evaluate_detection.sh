set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/official.txt}"
RUN_PREFIX="${2:-gate1-official}"
OUTPUT_FILE="${3:-logs/JBShield-D_runtime.log}"
JAILBREAKS="${4:-}"

./scripts/run_detection_config_list.sh "${LIST_FILE}" "${RUN_PREFIX}" "${OUTPUT_FILE}" "${JAILBREAKS}"
