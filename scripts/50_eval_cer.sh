#!/usr/bin/env bash
# scripts/50_eval_cer.sh
# 在 hyp.jsonl 上计算 CER（jiwer），输出总体 + 分方言两段报告。
# 用法：bash scripts/50_eval_cer.sh [run_name]
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-kespeech_run01}"
DECODE_DIR="exp/${RUN_NAME}/decode"
HYP="${DECODE_DIR}/hyp.jsonl"
REPORT="${DECODE_DIR}/cer_report.txt"

if [[ ! -f "${HYP}" ]]; then
  echo "[eval] missing ${HYP}; run 40_infer.sh first" >&2
  exit 1
fi

python - <<PY 2>&1 | tee "${REPORT}"
import json, re, unicodedata
from collections import defaultdict
from jiwer import process_characters, cer

PUNCT_RE = re.compile(r"[，。！？、；：“”‘’（）《》〈〉【】「」『』,.!?;:\"'()<>\[\]…—\-—_~\`@#\$%^&*+=|/\\\\]")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    s = PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s

def report(label, refs, hyps):
    if not refs:
        print(f"[{label}] (empty, skipped)")
        return
    out = process_characters(refs, hyps)
    total_chars = out.substitutions + out.deletions + out.hits  # ref_chars
    print(f"--- {label} ---")
    print(f"  num utterances : {len(refs)}")
    print(f"  hits           = {out.hits}")
    print(f"  substitutions  = {out.substitutions}")
    print(f"  deletions      = {out.deletions}")
    print(f"  insertions     = {out.insertions}")
    print(f"  ref_chars      = {total_chars}")
    print(f"  CER            = {cer(refs, hyps):.4f}")

# 读 hyp.jsonl，按 dialect 分桶
overall_refs, overall_hyps = [], []
by_dialect = defaultdict(lambda: ([], []))   # dialect -> (refs, hyps)
n_with_dialect = 0
with open("${HYP}", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        r = norm(ex.get("ref", ""))
        h = norm(ex.get("hyp", ""))
        if not r:
            continue
        overall_refs.append(r)
        overall_hyps.append(h)
        d = ex.get("dialect")
        if d:
            by_dialect[d][0].append(r)
            by_dialect[d][1].append(h)
            n_with_dialect += 1

print(f"=== CER report for run ${RUN_NAME} ===")
report("Overall", overall_refs, overall_hyps)

if by_dialect:
    print()
    print(f"=== Per-dialect CER (utt with dialect label: {n_with_dialect}) ===")
    rows = []
    for d, (refs, hyps) in sorted(by_dialect.items()):
        if not refs:
            continue
        c = cer(refs, hyps)
        rows.append((d, len(refs), c))
        report(f"dialect={d}", refs, hyps)
    print()
    print("--- Per-dialect summary table ---")
    print(f"{'dialect':<24} {'#utt':>8} {'CER':>10}")
    for d, n, c in sorted(rows, key=lambda x: x[2]):
        print(f"{d:<24} {n:>8} {c:>10.4f}")
else:
    print()
    print("(no dialect labels found in hyp.jsonl; per-dialect CER unavailable)")
PY

echo "[eval] report -> ${REPORT}"
