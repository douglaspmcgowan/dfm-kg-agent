#!/bin/bash
# Run Config C queries Q3-Q8 sequentially, each with a hard timeout enforced
# by GNU `timeout`. Append each result to benchmark_ablation_v2_C_only.json
# via a small Python merge script.
#
# Usage:  bash run_c_loop.sh

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

WORKER="benchmark_ablation_v2_C_worker.py"
OUT="benchmark_ablation_v2_C_only.json"
TIMEOUT_SEC=600   # 10 min per query

for Q in Q3 Q4 Q5 Q6 Q7 Q8; do
    TMP="tmp_${Q}_$$.json"
    TS_START="$(date +%H:%M:%S)"
    echo "[${TS_START}] === ${Q} starting (timeout=${TIMEOUT_SEC}s) ==="
    if timeout "${TIMEOUT_SEC}" python -u "$WORKER" "$Q" "$TMP" 4096 > "tmp_${Q}.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] ✓ ${Q} OK"
        # Merge the single-query record into the panel JSON
        python -c "
import json, sys
from pathlib import Path
out = Path('$OUT')
tmp = Path('$TMP')
rec = json.loads(tmp.read_text(encoding='utf-8'))
data = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {'records': [], 'model':'gpt-5', 'cap':4096}
data['records'] = [r for r in data['records'] if r.get('query_id') != rec.get('query_id')]
data['records'].append(rec)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'  saved {len(data[chr(34)records\"])} records')
"
    else
        EC=$?
        echo "[$(date +%H:%M:%S)] ✗ ${Q} failed (exit=${EC})"
        tail -10 "tmp_${Q}.log"
    fi
    rm -f "$TMP"
    echo ""
done

echo "ALL DONE at $(date +%H:%M:%S)"
