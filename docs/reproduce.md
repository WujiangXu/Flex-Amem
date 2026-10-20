# Reproducing the paper

Every arm shares the same budget accounting:
- **Budget:** a run fixes a token budget B for the whole prompt: the reader's prompt on LongMemEval and LoCoMo, the
  agent's prompt on SWE-bench and AppWorld. The cap is 0.95 B; the memory gets what the rest of the prompt leaves.
- **Shared rules:** the same units, embedder, token counter and overflow rule.
- **Memory resets:** memory starts empty for every question, issue or task.

The paper reports one run per arm. Before a full run, run one example end to end (`--limit 1` or `--ids <id>`).

## Arms and method keys

| Arm in the paper | LongMemEval-S, LoCoMo | SWE-bench Verified, AppWorld |
|---|---|---|
| FlexA-MEM | `flow_s30_ld50_noevo_zoom` | `flow_noevo_zoom` |
| FlexA-MEM with piece recovery (timeline + BM25 pieces) | `flow_s30_ld50_noevo_bm25` | `flow_noevo_bm25` |
| timeline only (no recovery) | `flow_s30_ld50_noevo_noexp` | `flow_noevo_zoom_noexp` |
| FlexA-MEM without excerpts (a unit is recovered only raw) | `flow_s30_ld50_noevo` | `flow_noevo` |
| A-MEM | `amem` | `amem` |
| A-MEM with fit-or-degrade rendering | `amem_zoom` | `amem_zoom` |
| Summary+A-MEM | `summary_amem_s50` | `summary_amem_s70` |
| ReadAgent | `readagent` | `readagent` |
| MemGAS | `memgas` | `memgas` |
| Sculptor | `sculptor` | `sculptor` |
| AFM (the paper's thresholds) | `afm` | `afm` |
| AFM, thresholds calibrated to MiniLM | `afm_t15_07` (LongMemEval), `afm_t30_15` (LoCoMo) | — |
| BM25 pieces, zero-build | `rag_pieces` | `rag_pieces` |
| raw-unit retrieval, zero-build | `rag_raw` | `rag_raw` |
| rule-based action trace | — | `trace` |
| observation masking | — | `masking` |
| observation masking + rule-based trace | — | `masking_trace` (AppWorld) |
| truncation | `truncation` | `truncation` |
| full history | `full` | `passthrough` |

Not every arm ran on every benchmark. The commands below list the arms of the paper's runs.

**Settings:**
- On LongMemEval and LoCoMo, FlexA-MEM gives the timeline a 0.3 share of the budget and discounts linked units by
  0.5 (`s30_ld50`).
- Excerpts are 300 tokens.
- Summary+A-MEM's history share (0.5 or 0.7) was chosen on the development splits.
- The method settings live in `src/flexamem/methods/` and in each runner's registry.

## LongMemEval-S (500 questions; 4k and 8k)
```bash
.venv/bin/python scripts/run_lme.py --run-id lme --split all --budgets 4000,8000 --workers 6 \
  --methods flow_s30_ld50_noevo_zoom,flow_s30_ld50_noevo_bm25,flow_s30_ld50_noevo,amem,amem_zoom,summary_amem_s50,readagent,memgas,sculptor,afm,afm_t15_07,rag_pieces,rag_raw,truncation,full
.venv/bin/python scripts/judge_lme.py --run-id lme
.venv/bin/python scripts/make_tables.py --lme-runs lme --flow-arm-lme flow_s30_ld50_noevo_zoom --out results/lme/tables
```
- `--split dev` is the 60-question development split, and `heldout` is the other 440.
- Each question's memory is built once per memory kind and reused across methods and budgets (`artifacts/lme_builds/`).
- **Builds in the paper vs here:** the paper's FlexA-MEM rows and the arms added later rendered the memories built by
  its first runs. `--builds <version>` reads such an existing build version. A fresh run builds with this code.

## LoCoMo (10 conversations, 1,540 questions of categories 1–4; 2k, 4k and 8k)
```bash
.venv/bin/python scripts/run_locomo.py --run-id locomo --budgets 2000,4000,8000 \
  --methods flow_s30_ld50_noevo_zoom,flow_s30_ld50_noevo_bm25,flow_s30_ld50_noevo_noexp,amem,amem_zoom,summary_amem_s50,readagent,memgas,sculptor,afm,afm_t30_15,rag_pieces,rag_raw,truncation,full
.venv/bin/python scripts/judge_locomo.py --run-id locomo
.venv/bin/python scripts/locomo_tables.py --runs locomo --out results/locomo/tables
```
- **Prompts:** the reader and the J judge use Mem0's prompts (`src/flexamem/bench/locomo/prompts.py`).
- **AFM calibration:** `afm_t30_15` was calibrated on conversation `conv-26`; `locomo_tables.py --exclude conv-26`
  reports it without that conversation.
- **Sculptor:** on LoCoMo, Sculptor's memory is rendered once and read by the ordinary reader, without the
  context-tool loop it gets on LongMemEval. This is how the paper's LoCoMo runs used it.

## SWE-bench Verified (Eval-95 and Rest-385; 8k)
```bash
scripts/pull_swe_images.sh eval95
.venv/bin/python scripts/run_swe.py --run-id swe_eval95 --split eval95 --budgets 8000 --workers 32 \
  --methods flow_noevo_zoom,flow_noevo,amem,summary_amem_s70,readagent,memgas,sculptor,afm,rag_pieces,rag_raw,masking,truncation,passthrough
.venv/bin/python scripts/scan_egress.py swe_eval95       # episodes that fetched external content (see below)
.venv/bin/python scripts/make_tables.py --swe-runs swe_eval95 --flow-arm-swe flow_noevo_zoom --out results/swe/tables
```
- **Agent:** mini-swe-agent's SWE-bench configuration (step limit 250), with the problem statement alone as the task.
- **Grading:** the official harness grades each patch in a fresh container.
- **Errors:** infrastructure failures leave no row and are retried when the run resumes. A method error is recorded
  as unresolved.
- **Rest-385:** `--split rest385`.
- **Offline reruns:** a runner skips episodes that already have a row, and the tables refuse an episode found in two
  runs. To rerun a flagged episode offline:
  1. move its row out of `artifacts/runs/swe/<run_id>/results.jsonl` (keep a copy);
  2. run `run_swe.py` again on the same run id with `--no-egress --ids <instance> --methods <arm> --budgets <B>`
     (`--allow-mixed` if the code has changed since).
- **Tables:** `make_tables.py --swe-runs` expects the arm set above. Its pre-registered endpoint tables stop if the
  runs hold no Summary+A-MEM rows at 8k.

**Checkpoint diagnostic.** This is the recall probe at fixed points of SWE trajectories. It needs a passthrough run
that saved its checkpoints:
```bash
.venv/bin/python scripts/run_swe.py --run-id swe_pass --split eval95 --methods passthrough --snapshots
.venv/bin/python scripts/run_diag.py --run-id diag --source-run swe_pass --budget 8000 \
  --methods flow_noevo_zoom,amem,summary_amem_s70,memgas,readagent,masking,truncation,passthrough
.venv/bin/python scripts/make_tables.py --diag-runs diag --flow-arm-swe flow_noevo_zoom --out results/diag/tables
```

## AppWorld (Test-Normal 168 tasks, Test-Challenge 417; 4k and 6k)
```bash
.venv/bin/python scripts/run_appworld.py --run-id aw_normal --split test_normal --budgets 4000,6000 --workers 40 \
  --methods flow_noevo_zoom,flow_noevo_bm25,flow_noevo_zoom_noexp,flow_noevo,amem,amem_zoom,summary_amem_s70,readagent,memgas,sculptor,afm,rag_pieces,rag_raw,trace,masking,masking_trace,truncation,passthrough
.venv/bin/python scripts/scan_appworld_leak.py aw_normal
.venv/bin/python scripts/appworld_tables.py --runs aw_normal --out results/aw_normal/tables
```
- **Agent loop:** SWE's, with a `python` action tool in the task's world, AppWorld's instructions without few-shot
  demonstrations, and a step limit of 60.
- **Other splits:**
  - `--split test_challenge` runs the harder split;
  - `--split train,dev` runs the confirmation set, which the runner restricts to `configs/splits/appworld_confirm.txt`;
  - `--shard i/n` lets n processes share one run id.

## Statistics
Tables go wherever `--out` points (`results/` is git-ignored). `configs/exp/endpoints.yaml` names the pre-registered
reference arms of the endpoint tests (`flow`, `flow_s30_ld50`). For the paper's comparisons against FlexA-MEM, pass
`--flow-arm-swe flow_noevo_zoom --flow-arm-lme flow_s30_ld50_noevo_zoom`, as in the commands above.
`appworld_tables.py` and `locomo_tables.py` compare against FlexA-MEM by default (`--ours`).

The table scripts report paired contrasts against FlexA-MEM:
- exact McNemar tests;
- bootstrap intervals clustered by repository, conversation or AppWorld scenario;
- TOST for non-inferiority.

The implementations are in `src/flexamem/analysis/stats.py`.
