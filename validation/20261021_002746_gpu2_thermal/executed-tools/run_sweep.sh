#!/usr/bin/env bash
# Concurrency sweep of stream_bench.py with a fresh server for every run, as
# PR #8933 measured (a new server per concurrency level; here per repeat too).
#
# Each run: start SERVER_CMD in its own process group, wait for /health, log
# GPU clocks / power / throttle reasons, benchmark, stop the server (SIGINT,
# then SIGTERM, then SIGKILL), and wait until the GPU memory and the port are
# released before the next run.
#
# Required:
#   SERVER_CMD  starts the server in the foreground, e.g.
#               'source ~/activate_cuda129.sh && CUDA_VISIBLE_DEVICES=1 vllm serve /models/drama-3 --omni --port 8091'
#   CODES       JSON Lines from encode_ref_codes.py
# Optional:
#   PORT=8091  CONCURRENCY="1 16 32 48 64 96 128"  REPEATS=1
#   REPEATS_AT="64:3 128:5"  per-level repeats (PR #8933 averaged 3 runs at 64, 5 at 128)
#   OUT=results/sweep_<time>  LABEL=<run label>  GPU_INDEX=0 (nvidia-smi index or UUID of the served GPU)
#   READY_TIMEOUT=1800  STOP_TIMEOUT=120  GPU_IDLE_MIB=1024
#   BENCH_ARGS="..."  extra stream_bench.py flags (e.g. "--save-audio")
#   CLIENT_PREFIX="taskset -c 60-63"  keeps the client off the server's cores
#   PYTHON=python3
set -euo pipefail

: "${SERVER_CMD:?set SERVER_CMD (see the header of this script)}"
: "${CODES:?set CODES to the encode_ref_codes.py output}"
PORT=${PORT:-8091}
CONCURRENCY=${CONCURRENCY:-"1 16 32 48 64 96 128"}
REPEATS=${REPEATS:-1}
REPEATS_AT=${REPEATS_AT:-}
OUT=${OUT:-results/sweep_$(date +%Y%m%d_%H%M%S)}
LABEL=${LABEL:-}
GPU_INDEX=${GPU_INDEX:-0}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
STOP_TIMEOUT=${STOP_TIMEOUT:-120}
GPU_IDLE_MIB=${GPU_IDLE_MIB:-1024}
BENCH_ARGS=${BENCH_ARGS:-}
CLIENT_PREFIX=${CLIENT_PREFIX:-}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/sweep.log"; }

repeats_for() {
  local level=$1 entry
  for entry in $REPEATS_AT; do
    if [[ ${entry%%:*} == "$level" ]]; then echo "${entry#*:}"; return; fi
  done
  echo "$REPEATS"
}

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; }

gpu_mem_mib() {
  nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

# Throttle reasons were renamed clocks_event_reasons in newer drivers.
GPU_FIELDS="timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used"
if nvidia-smi -i "$GPU_INDEX" --query-gpu=clocks_event_reasons.active --format=csv,noheader >/dev/null 2>&1; then
  GPU_FIELDS+=",clocks_event_reasons.active"
else
  GPU_FIELDS+=",clocks_throttle_reasons.active"
fi

SERVER_PGID=""
MONITOR_PID=""
cleanup() {
  [[ -n $MONITOR_PID ]] && kill "$MONITOR_PID" 2>/dev/null || true
  if [[ -n $SERVER_PGID ]] && kill -0 -- "-$SERVER_PGID" 2>/dev/null; then
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stop_server() {
  local run_dir=$1 sig waited
  for sig in INT TERM KILL; do
    kill -0 -- "-$SERVER_PGID" 2>/dev/null || break
    kill "-$sig" -- "-$SERVER_PGID" 2>/dev/null || true
    for ((waited = 0; waited < STOP_TIMEOUT; waited++)); do
      kill -0 -- "-$SERVER_PGID" 2>/dev/null || break
      sleep 1
    done
    [[ $sig != INT ]] && log "server needed SIG$sig" && echo "SIG$sig" >>"$run_dir/stop_signals.txt"
  done
  SERVER_PGID=""
  for ((waited = 0; waited < STOP_TIMEOUT; waited++)); do
    if ! port_open && (($(gpu_mem_mib) < GPU_IDLE_MIB)); then
      echo "{\"gpu_mem_after_mib\": $(gpu_mem_mib), \"wait_s\": $waited}" >"$run_dir/cleanup.json"
      return 0
    fi
    sleep 1
  done
  log "GPU memory ($(gpu_mem_mib) MiB) or port $PORT not released after ${STOP_TIMEOUT}s"
  return 1
}

if port_open; then
  log "port $PORT is already in use; stop that server first"
  exit 1
fi
LABEL="$LABEL" CONCURRENCY="$CONCURRENCY" REPEATS="$REPEATS" REPEATS_AT="$REPEATS_AT" CODES="$CODES" \
  PORT="$PORT" GPU_INDEX="$GPU_INDEX" BENCH_ARGS="$BENCH_ARGS" SERVER_CMD="$SERVER_CMD" "$PYTHON" - >"$OUT/sweep.json" <<'PY'
import json, os, socket, time
keys = ["LABEL", "CONCURRENCY", "REPEATS", "REPEATS_AT", "CODES", "PORT", "GPU_INDEX", "BENCH_ARGS", "SERVER_CMD"]
info = {k.lower(): os.environ[k] for k in keys}
info.update(codes=os.path.realpath(info["codes"]), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
info["hostname"] = socket.gethostname()
print(json.dumps(info, indent=2))
PY
printf '%s\n' "$SERVER_CMD" >"$OUT/server_cmd.txt"

for level in $CONCURRENCY; do
  for ((rep = 1; rep <= $(repeats_for "$level"); rep++)); do
    run_dir="$OUT/c${level}/r${rep}"
    mkdir -p "$run_dir"
    log "c=$level r=$rep: starting server"
    nvidia-smi -i "$GPU_INDEX" --query-gpu="$GPU_FIELDS" --format=csv -lms 500 >"$run_dir/gpu.csv" 2>&1 &
    MONITOR_PID=$!
    t0=$(date +%s)
    setsid bash -c "$SERVER_CMD" >"$run_dir/server.log" 2>&1 &
    SERVER_PGID=$!
    pgid=$(ps -o pgid= -p "$SERVER_PGID" | tr -d ' ' || true)
    if [[ -n $pgid && $pgid != "$SERVER_PGID" ]]; then
      log "c=$level r=$rep: server is not its own process group; stopping it may leave processes behind"
    fi
    until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
      if ! kill -0 "$SERVER_PGID" 2>/dev/null; then
        log "c=$level r=$rep: server exited during startup, see $run_dir/server.log"
        exit 1
      fi
      if (($(date +%s) - t0 > READY_TIMEOUT)); then
        log "c=$level r=$rep: not ready after ${READY_TIMEOUT}s"
        stop_server "$run_dir" || true
        exit 1
      fi
      sleep 2
    done
    echo $(($(date +%s) - t0)) >"$run_dir/startup_seconds.txt"
    log "c=$level r=$rep: ready after $(cat "$run_dir/startup_seconds.txt")s, benchmarking"
    set +e
    # shellcheck disable=SC2086 # CLIENT_PREFIX and BENCH_ARGS are word lists
    $CLIENT_PREFIX "$PYTHON" "$HERE/stream_bench.py" --codes "$CODES" --concurrency "$level" \
      --out "$run_dir" --base-url "$BASE_URL" --label "$LABEL" \
      --meta "repeat=$rep" --meta "startup_s=$(cat "$run_dir/startup_seconds.txt")" \
      $BENCH_ARGS 2>&1 | tee "$run_dir/client.log"
    status=${PIPESTATUS[0]}
    set -e
    log "c=$level r=$rep: client exit $status, stopping server"
    stop_server "$run_dir" || exit 1
    kill "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  done
done

"$PYTHON" "$HERE/aggregate_results.py" "$OUT" --json "$OUT/table.json" | tee "$OUT/table.md"
log "done: $OUT/table.md"
