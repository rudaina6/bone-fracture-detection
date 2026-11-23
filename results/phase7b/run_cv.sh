#!/usr/bin/env bash
# Phase 7b CV driver: one step at a time, stdout+stderr per step to results/phase7b/stage3_<step>.log,
# one commit after each step (no push). Stops at the first failing step.
#   bash results/phase7b/run_cv.sh
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=results/phase7b
NJ=3

step() {  # step <name> <commit message> <cascade args...>
  local name=$1 msg=$2; shift 2
  local log=$OUT/stage3_$name.log
  echo "[$(date '+%F %T')] start $name: $*" | tee -a $OUT/stage3_driver.log
  $PY src/phase7b_cascade.py "$@" --n-jobs $NJ > "$log" 2>&1
  local rc=$?
  echo "exit $rc" >> "$log"
  echo "[$(date '+%F %T')] end $name exit $rc" | tee -a $OUT/stage3_driver.log
  if [ $rc -ne 0 ]; then echo "STEP $name FAILED" | tee -a $OUT/stage3_driver.log; exit $rc; fi
  git add $OUT results/tables results/figures
  git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "(nothing to commit for $name)"
}

step stage1 "Phase 7b CV: stage 1 (cross-fitted round-0 SVMs, top-1000 proposals)" --stage1
for k in 0 1 2 3 4; do
  step cv_fold$k "Phase 7b CV: fold $k (4 variants, held-out detections)" --cv-fold $k
done
step select "Phase 7b CV: variant selection (B1-B3)" --select
step final_params "Phase 7b CV: final hyperparameters on all training images" --final-params
NEED=$($PY -c "import sys; sys.path.insert(0,'src'); import argparse, pathlib, phase7b_cascade as C; \
a=argparse.Namespace(out_dir=pathlib.Path('$OUT'), p7_dir=pathlib.Path('results/phase7')); n,i=C.threshold_needed(a); print(int(n)); print(i, file=sys.stderr)" 2>>$OUT/stage3_driver.log)
echo "threshold CV needed: $NEED" | tee -a $OUT/stage3_driver.log
if [ "$NEED" = "1" ]; then
  step threshold_cv "Phase 7b CV: threshold CV with the final hyperparameters (B4)" --threshold-cv 0 1 2 3 4
fi
step summarize "Phase 7b CV: summary (B4 threshold, per-region metrics, comparisons, B6)" --summarize
echo "[$(date '+%F %T')] ALL DONE" | tee -a $OUT/stage3_driver.log
