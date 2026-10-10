#!/bin/bash
# Run Config C queries Q3-Q8 sequentially via GNU timeout. Keep tmp result
# files (don't delete them) so any merge failure doesn't lose data.
#
# Usage:  bash run_c_loop2.sh

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

WORKER="benchmark_ablation_v2_C_worker.py"
MERGER="merge_c_result.py"
OUT="benchmark_ablation_v2_C_only.json"
TIMEOUT_SEC=600   # 10 min per query

for Q in Q3 Q4 Q5 Q6 Q7 Q8; do
    TMP="result_${Q}.json"
    TS_START="$(date +%H:%M:%S)"
    echo "[${TS_START}] === ${Q} starting (timeout=${TIMEOUT_SEC}s) ==="
    if timeout "${TIMEOUT_SEC}" python -u "$WORKER" "$Q" "$TMP" 4096 > "tmp_${Q}.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] OK ${Q}"
        if [ -f "$TMP" ]; then
            python "$MERGER" "$TMP" "$OUT"
        else
            echo "[$(date +%H:%M:%S)] WARN ${Q} exit 0 but no $TMP"
        fi
    else
        EC=$?
        echo "[$(date +%H:%M:%S)] FAIL ${Q} (exit=${EC})"
        echo "--- worker stdout tail ---"
        tail -5 "tmp_${Q}.log"
        echo "---"
    fi
    echo ""
done

echo "ALL DONE at $(date +%H:%M:%S)"
