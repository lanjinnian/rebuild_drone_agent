#!/usr/bin/env bash
set -euo pipefail

CHECK_INTERVAL=60
MIN_FREE_MEMORY_MB=0
REQUIRED_FREE_CHECKS=1

usage() {
  cat <<'EOF'
Usage:
  scripts/run_when_gpu_free.sh [options] -- <command> [args...]

Options:
  -i, --interval <seconds>       GPU check interval. Default: 60
  -m, --min-free-mem <MB>        Required free GPU memory. Default: 0
  -n, --free-checks <count>      Required consecutive free checks. Default: 1
  -h, --help                     Show this help.

Example:
  scripts/run_when_gpu_free.sh -i 30 -n 3 -- python main.py data/example/01.mp4

When a GPU has no compute process, the script sets CUDA_VISIBLE_DEVICES to that
GPU index and runs the command. With -n, the same GPU must be free for that many
consecutive checks before the command starts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval)
      CHECK_INTERVAL="$2"
      shift 2
      ;;
    -m|--min-free-mem)
      MIN_FREE_MEMORY_MB="$2"
      shift 2
      ;;
    -n|--free-checks)
      REQUIRED_FREE_CHECKS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing command to run." >&2
  usage >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found." >&2
  exit 1
fi

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_positive_integer "$CHECK_INTERVAL" || [[ "$CHECK_INTERVAL" -eq 0 ]]; then
  echo "--interval must be a positive integer." >&2
  exit 2
fi

if ! is_positive_integer "$MIN_FREE_MEMORY_MB"; then
  echo "--min-free-mem must be a non-negative integer." >&2
  exit 2
fi

if ! is_positive_integer "$REQUIRED_FREE_CHECKS" || [[ "$REQUIRED_FREE_CHECKS" -eq 0 ]]; then
  echo "--free-checks must be a positive integer." >&2
  exit 2
fi

gpu_is_free() {
  local gpu_index="$1"
  local used_by_compute
  local free_memory

  used_by_compute="$(
    nvidia-smi \
      --query-compute-apps=gpu_bus_id \
      --format=csv,noheader,nounits \
      -i "$gpu_index" 2>/dev/null \
      | sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "$used_by_compute" ]]; then
    return 1
  fi

  free_memory="$(
    nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      -i "$gpu_index" \
      | head -n 1 \
      | tr -d '[:space:]'
  )"
  [[ "$free_memory" -ge "$MIN_FREE_MEMORY_MB" ]]
}

gpu_count="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l
)"

if [[ "$gpu_count" -eq 0 ]]; then
  echo "No NVIDIA GPU found." >&2
  exit 1
fi

declare -a free_counts
for gpu_index in $(seq 0 "$((gpu_count - 1))"); do
  free_counts["$gpu_index"]=0
done

echo "Waiting for a free GPU. interval=${CHECK_INTERVAL}s min_free_mem=${MIN_FREE_MEMORY_MB}MB free_checks=${REQUIRED_FREE_CHECKS}"

while true; do
  for gpu_index in $(seq 0 "$((gpu_count - 1))"); do
    if gpu_is_free "$gpu_index"; then
      free_counts["$gpu_index"]=$((free_counts["$gpu_index"] + 1))
      echo "$(date '+%F %T') GPU ${gpu_index} is free (${free_counts["$gpu_index"]}/${REQUIRED_FREE_CHECKS})"
      if [[ "${free_counts["$gpu_index"]}" -ge "$REQUIRED_FREE_CHECKS" ]]; then
        echo "GPU ${gpu_index} stayed free. Running: $*"
        export CUDA_VISIBLE_DEVICES="$gpu_index"
        exec "$@"
      fi
    else
      free_counts["$gpu_index"]=0
    fi
  done

  echo "$(date '+%F %T') waiting, checking again in ${CHECK_INTERVAL}s"
  sleep "$CHECK_INTERVAL"
done
