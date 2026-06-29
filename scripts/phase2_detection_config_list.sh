#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/official.txt}"
RUN_PREFIX="${2:-phase2-official}"
OUTPUT_FILE="${3:-logs/JBShield-D_phase2.log}"
JAILBREAKS="${4:-}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"
: > "${OUTPUT_FILE}"

JBSHIELD_CLEAN_HF_CACHE="${JBSHIELD_CLEAN_HF_CACHE:-always}"
CURRENT_CONFIG=""

cleanup_hf_cache_for_config() {
    local config="$1"
    local reason="$2"
    local policy="${JBSHIELD_CLEAN_HF_CACHE:-never}"
    local should_cleanup="false"

    case "${policy}" in
        always|true|1|yes)
            should_cleanup="true"
            ;;
        on_success)
            [[ "${reason}" == "success" ]] && should_cleanup="true"
            ;;
        on_failure|interrupted)
            [[ "${reason}" == "failure" || "${reason}" == "interrupted" ]] && should_cleanup="true"
            ;;
        never|false|0|no|"")
            should_cleanup="false"
            ;;
        *)
            echo "Warning: unknown JBSHIELD_CLEAN_HF_CACHE='${policy}', skip cache cleanup." >> "${OUTPUT_FILE}"
            should_cleanup="false"
            ;;
    esac

    if [[ "${should_cleanup}" != "true" ]]; then
        return 0
    fi

    local cleanup_args=(--config "${config}")
    if [[ "${JBSHIELD_CLEAN_HF_CACHE_DRY_RUN:-0}" == "1" ]]; then
        cleanup_args+=(--dry-run)
    fi

    echo "===== HF cache cleanup (${reason}): ${config} =====" >> "${OUTPUT_FILE}"
    if ! uv run python -u scripts/cleanup_hf_cache.py "${cleanup_args[@]}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Warning: HF cache cleanup failed for ${config}" >> "${OUTPUT_FILE}"
    fi
}

cleanup_current_config_on_exit() {
    if [[ -n "${CURRENT_CONFIG}" ]]; then
        cleanup_hf_cache_for_config "${CURRENT_CONFIG}" "interrupted"
    fi
}
trap cleanup_current_config_on_exit EXIT INT TERM

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
    CURRENT_CONFIG="${CONFIG}"
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
        echo "Skip ${CONFIG}: Phase2 detection failed. Check GPU memory, model files, gated model access, HF download/cache state, span mapping, or model-specific runtime support." >> "${OUTPUT_FILE}"
        cleanup_hf_cache_for_config "${CONFIG}" "failure"
        CURRENT_CONFIG=""
        continue
    fi
    cleanup_hf_cache_for_config "${CONFIG}" "success"
    CURRENT_CONFIG=""
done < "${LIST_FILE}"
