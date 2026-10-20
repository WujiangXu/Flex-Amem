# FlexA-MEM: Budget-Adaptive Agentic Memory for LLM Agents

This repository holds the code and evaluation harness for the FlexA-MEM paper.

An LLM agent's history outgrows its context window. FlexA-MEM keeps the whole history in one linked memory:
- **Writing:** each unit of history gets a short note from an LLM, and the notes form a timeline.
- **Rendering under a token budget:** the agent sees the timeline plus the details of the units most relevant to the
  current step. A unit is shown raw when it fits and as a short excerpt when it does not ("fit or degrade").

The memory therefore both compresses the history and indexes how to recover its details. It is evaluated on four
benchmarks: LongMemEval-S, LoCoMo, SWE-bench Verified and AppWorld. The baselines are A-MEM, Summary+A-MEM,
ReadAgent, MemGAS, Sculptor and AFM, plus zero-build retrieval, observation masking, truncation and full history.
All arms share one token counter, embedder, budget and overflow rule.

## Quick start
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock   # installs this package (-e .)
export OPENAI_API_KEY=...            # and OPENAI_BASE_URL=... for another OpenAI-compatible endpoint
scripts/download_data.sh             # LongMemEval-S, SWE-bench Verified, LoCoMo (sha256-checked)

# one LongMemEval question with FlexA-MEM at a 4k-token budget, then the judges
# (--allow-dirty: runs record the commit and otherwise refuse an uncommitted tree; the judges need an endpoint
#  that serves the judge models of configs/models/default.yaml, see docs/setup.md)
.venv/bin/python scripts/run_lme.py --run-id try_lme --split all --limit 1 \
    --methods flow_s30_ld50_noevo_zoom --budgets 4000 --allow-dirty
.venv/bin/python scripts/judge_lme.py --run-id try_lme
```
Model roles live in `configs/models/default.yaml`. Every LLM call goes through `src/flexamem/llm/client.py`, which logs
it to `artifacts/ledger/`.

## Documentation
- **Setup:** `docs/setup.md` covers models, data, the SWE-bench images and AppWorld.
- **Reproduction:** `docs/reproduce.md` lists the paper's arms with their method keys, plus commands and tables for
  every benchmark.

In the code, FlexA-MEM is the method `flow_noevo_zoom`, and `flow_s30_ld50_noevo_zoom` on the conversational
benchmarks. Method keys keep the names of the experiments.

## Layout
```
configs/   models/ (roles), exp/ (table settings), splits/ (frozen id lists with sha256 manifests)
scripts/   run_lme.py judge_lme.py run_locomo.py judge_locomo.py run_swe.py run_diag.py run_appworld.py
           make_tables.py locomo_tables.py appworld_tables.py scan_egress.py scan_appworld_leak.py
           download_data.sh pull_swe_images.sh setup_appworld.sh
src/flexamem/
  llm/        client (ledger, cache, retries), tokens, embed, agent_model (mini-swe-agent model)
  memory/     segmenter, store, writer, linker, merge, prompts
  render/     budgeted rendering for every method, excerpts and BM25 pieces
  methods/    flow (FlexA-MEM), amem (+ Summary+A-MEM), readagent, memgas, sculptor, afm, rag, trace, hybrid
  adapters/   mini-swe-agent context policies (memory, masking, masking+trace, truncation, passthrough)
  bench/      longmemeval/, locomo/, swe/, appworld/
  analysis/   statistics (McNemar, clustered bootstrap, TOST), tables, token cost
```
Code comments sometimes cite a "design deviation", a "decision", the "journal", or documents under `docs/` and
`results/` of the research repository. These belong to the project's research log, which is not part of this
release.

## License
MIT (`LICENSE`). `NOTICE` lists the code, prompts and software this release reuses; `licenses/` holds their license texts.
