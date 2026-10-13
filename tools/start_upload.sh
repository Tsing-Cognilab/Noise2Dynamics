#!/usr/bin/env bash
set -uo pipefail
cd /data/lizhenhao/lizhenhao/Noise2Dynamics_CVM || exit 1
python3 -u tools/publish.py --target both 2>&1 | tee -a upload-dual.log
cvm_status=${PIPESTATUS[0]}
printf '\nUploader exit code: %s\n' "$cvm_status"
if [ "$cvm_status" -ne 0 ]; then
    printf 'Stopped; verified chunks are retained. Rerun: python3 -u tools/publish.py --target both\n'
fi
exec bash
