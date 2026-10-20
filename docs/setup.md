# Setup

## Environment
Python 3.12. `requirements.lock` pins every package of the environment that ran the paper's experiments, among them
mini-swe-agent 2.3.0, swebench 4.1.0, litellm 1.82.6, openai 2.33.0, sentence-transformers and tiktoken 0.13.0:
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock
```
The lock pins Linux x86-64 wheels, including a CUDA build of torch (several GB), although the embedder runs on CPU.

Generated files go to `artifacts/` (git-ignored):

| What | Where |
|---|---|
| data | `artifacts/data/` |
| runs | `artifacts/runs/<benchmark>/<run_id>/results.jsonl` |
| per-call ledger | `artifacts/ledger/` |
| LLM cache | `artifacts/cache/llm_cache.sqlite` |

**Git state.** Every run records the code's commit:
- A runner refuses an uncommitted tree unless you pass `--allow-dirty`.
- Resuming a run whose rows came from another commit needs `--allow-mixed`.
- `results/` and `logs/` are git-ignored. Keep your own model config outside the repository (`FLEXAMEM_MODELS`).

## Models
`configs/models/default.yaml` assigns a model to each role:

| Role | Used for |
|---|---|
| `agent` | SWE-bench and AppWorld agent |
| `memory` | every memory-side call: note writing, linking, merging, summarizing, gists, lookups |
| `reader` | answers on LongMemEval and LoCoMo |
| `judge`, `judge2`, `judge_locomo` | answer checking |

In the paper's runs:
- GPT-5.4 mini filled the agent, memory and reader roles.
- The judges came from other families: Claude Sonnet 4.6 and GPT-5.4.

**Endpoint and key.** The client speaks the OpenAI chat-completions API. It takes:
- the endpoint from `base_url` in the config or from `OPENAI_BASE_URL` (default: the OpenAI API);
- the key from `OPENAI_API_KEY`.

To use another file, set `FLEXAMEM_MODELS=<path>`. A file under `configs/models/` given as
`FLEXAMEM_ROLES=configs/models/<file>.yaml` replaces only the roles it lists (e.g. a second agent backbone).

**Model availability.**
- **Judges:** the judge roles need an endpoint that serves Claude Sonnet 4.6 and GPT-5.4, e.g. a gateway with an
  OpenAI-compatible API. Other judges work but change comparability with the paper.
- **Agent calls on the OpenAI API:** for GPT-5.4 and later models with tools and a reasoning effort, litellm sends
  agent turns through the Responses API, because chat completions refuses that combination there. The paper's
  gateway accepted them as chat completions.
- **Sculptor's reader:** on LongMemEval and LoCoMo it calls tools with a reasoning effort through chat completions, so
  it needs an endpoint that accepts that.

**Token counting, embeddings and caching:**
- **Token counter:** budgets are counted with tiktoken `o200k_base` over every message field and the tool schema.
  tiktoken downloads the file on first use.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` at Hugging Face revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, the revision of every paper run (pinned in `src/flexamem/llm/embed.py`).
- **Cache:** identical memory-side prompts are answered from the LLM cache. The ledger keeps each call's original
  usage, so token costs are reported as if nothing had been cached. Set `FLEXAMEM_LLM_CACHE=<path>` to give a run
  its own cache.
- **What the cache and builds are keyed by:**
  - **Cache:** keyed by model name and prompt, not by endpoint.
  - **Memory builds:** keyed by the code version (`artifacts/lme_builds/`, `artifacts/locomo_builds/`), not by the
    memory model.
  - **When you change endpoint or models:** use a separate cache and remove (or move) the builds.

## Data
```bash
scripts/download_data.sh
```
It downloads three datasets at pinned revisions and checks their sha256:
- LongMemEval-S (cleaned): Hugging Face `xiaowu0162/longmemeval-cleaned`;
- SWE-bench Verified: `princeton-nlp/SWE-bench_Verified`;
- LoCoMo: `locomo10.json`.

The split files in `configs/splits/` are the frozen id lists used in the paper:
- LongMemEval: the 60-question development split `lme_dev60.txt`;
- SWE-bench Verified: Dev20 and Eval-100. Eval-95 and Rest-385 are Eval-100 and the remaining instances after
  removing those whose gold patch fails;
- AppWorld: the confirmation set.

`manifest*.json` holds their sha256.

The datasets are not part of this repository; each comes under its own terms:
- LongMemEval and its cleaned release: the dataset card;
- LoCoMo: snap-research/locomo;
- SWE-bench Verified: princeton-nlp/SWE-bench_Verified;
- AppWorld: its repository.

## SWE-bench Verified images
The agent runs in the SWE-bench 4.1.0 instance image of each task. Runs never pull images (`--pull=never`), so pull
a split's images first:
```bash
scripts/pull_swe_images.sh eval95      # or rest385, dev20, eval100
```
- **Docker:** a running daemon is required (Docker Engine 25 or newer: the grader pins API version 1.44). Grading
  uses the daemon that `docker.from_env()` finds (`DOCKER_HOST`, else the default socket).
- **How the agent runs:** in its container on the host network. `run_swe.py --no-egress` starts containers with
  `--network=none` instead. `scripts/scan_egress.py <run_id>` lists episodes whose agent fetched external content;
  in the paper, those episodes were rerun with `--no-egress`.
- **Image differences:** the paper's runs used a mirror of the SWE-bench 4.1.0 images. Its environments set proxy
  variables to an unreachable local proxy, so tests had no internet access, and a few instances' environments differ
  from the Docker Hub builds. The gold-patch exclusions behind Eval-95 and Rest-385 were measured on those images.
  On other images, check the gold patches of the pulled split first. The harness writes `logs/` and a report into
  the current directory:
  ```bash
  .venv/bin/python -m flexamem.bench.swe.grade --dataset_name princeton-nlp/SWE-bench_Verified \
      --predictions_path gold --run_id gold_check --max_workers 8 \
      --instance_ids $(cat configs/splits/swe_verified_eval95.txt)
  ```

## AppWorld
AppWorld pins pydantic<2, so it runs in its own environment:
```bash
scripts/setup_appworld.sh     # artifacts/appworld/venv + data in artifacts/appworld/root
```
Each episode starts `src/flexamem/bench/appworld/bridge.py` in that environment and talks to it over JSON lines.
AppWorld's evaluator scores the final state after the episode. The ground truth never reaches the agent or its
memory, and `scripts/scan_appworld_leak.py <run_id>` checks the trajectories for code that touched it.
