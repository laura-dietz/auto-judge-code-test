#!/bin/bash
#
# meta-watch.sh - Sync, meta-evaluate, and publish leaderboard correlations
#

set -euo pipefail


# === CONFIGURE THESE ===

# Rsync source and destination (parent directories)
RSYNC_SRC="server:/auto-judge/in"  # Change this where auto-judge results are collected
RSYNC_DEST="server:/auto-judge/out/"  # Change this to where correlation result should get written to

TRUTH="auto-judge/truth-eval" # Change this to directory with official leaderboards



# Local directories
WATCH_PARENT="./eval-watch/in"
OUT_PARENT="./eval-watch/out"

# Shared options
COMMON_OPTS=" --only-shared-runs   --truth-drop-aggregate --out-format jsonl  --only-shared-topics  --on-missing skip"
COMMON_OPTS=" --only-shared-runs   --truth-drop-aggregate --out-format jsonl  --all-topics  --on-missing skip"


CORR_OPTS=" --correlation kendall  --correlation pearson --correlation spearman  --correlation kendall --correlation tauap_b --correlation kendall@15   --correlation spearman@15 --correlation kendall@5   --correlation spearman@5"
# === END CONFIG ===

mkdir -p "$WATCH_PARENT" "$OUT_PARENT"
mkdir -p "$OUT_PARENT/ragtime" "$OUT_PARENT/rag" "$OUT_PARENT/rag-auggen" "$OUT_PARENT/dragun"

TS=$(date +%F_%H-%M-%S)

# Sync incoming
# rsync -Laura "$RSYNC_SRC/" "$WATCH_PARENT/" # 2>/dev/null || true
CHANGE_COUNT=$(rsync -Laurai "$RSYNC_SRC/" "$WATCH_PARENT/" 2>/dev/null | wc -l) 
echo "change count $CHANGE_COUNT"

if [ "$CHANGE_COUNT" -gt 0 ]; then  
    LOG="$OUT_PARENT/log/correlation-$TS.log"
    mkdir -p "$OUT_PARENT/log"

    # echo "Logging to $LOG"

    set -x

    trec-auto-judge meta-evaluate  --truth-leaderboard $TRUTH/ragtime-export/eval/ragtime.repgen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/ragtime/*eval.txt \
    $CORR_OPTS $COMMON_OPTS \
    --truth-measure nugget_coverage --truth-measure correct_nuggets  --truth-measure f1  \
    --output $OUT_PARENT/ragtime/correlations-$TS.jsonl  #>> $LOG 2>&1


    trec-auto-judge meta-evaluate  --truth-leaderboard $TRUTH/rag-export/eval/rag.generation.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/rag/*eval.txt \
    $CORR_OPTS $COMMON_OPTS \
    --output $OUT_PARENT/rag/correlations-$TS.jsonl #>> $LOG  2>&1 
    
    trec-auto-judge meta-evaluate  --truth-leaderboard $TRUTH/rag-export/eval/rag.auggen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/rag/*eval.txt \
    $CORR_OPTS $COMMON_OPTS \
    --output $OUT_PARENT/rag-auggen/correlations-$TS.jsonl  # >> $LOG 2>&1

    trec-auto-judge meta-evaluate  --truth-leaderboard $TRUTH/dragun-export/eval/dragun.repgen.official.eval.jsonl --truth-format jsonl  \
    --eval-format tot  -i $WATCH_PARENT/dragun/*eval.txt \
    $CORR_OPTS $COMMON_OPTS \
    --output $OUT_PARENT/dragun/correlations-$TS.jsonl #>> $LOG 2>&1
    

    # Publish results
    rsync -Laura "$OUT_PARENT/" "$RSYNC_DEST/" # 2>/dev/null || true
# else echo "no change"
fi

