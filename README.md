# FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents 🧠

A budget-adaptive agentic memory that lets one linked memory serve as both the compressed history and the index for
recovering its details, under any token budget.

> **Note:** This repository holds the code and evaluation harness to reproduce the results in our paper. FlexA-Mem
> builds on our earlier work [A-Mem](https://github.com/WujiangXu/A-mem) and gives a stronger, budget-adaptive version
> of it.

For more details, please refer to our paper: [FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents](https://arxiv.org/html/2610.33274v1)

## Introduction 🌟

An LLM agent's history soon outgrows its context window. The usual fix pairs a summarizer with a separate retriever.
The summarizer keeps a running account of the past but drops its details. The retriever recovers details from a
separate index and appends them to the prompt.

FlexA-Mem tests whether **one linked memory can do both jobs**. Each unit of history is written once as an A-Mem-style
note with links to related notes, and the raw unit is kept unchanged. At every step the agent sees the whole history as
a timeline of short notes. The units most relevant to the current step appear in place with their details: a unit is
shown raw when it fits the budget and as a short excerpt when it does not ("fit or degrade"). The memory therefore
compresses the history and indexes how to recover its details, and the prompt always fits the budget.

## Key Features ✨

- 🧾 One linked memory serves as both the compressed history and the retrieval index
- 📏 Budget-adaptive rendering: one memory serves any token budget without rebuilding
- 🔍 Details are recovered in place through vector similarity and one-hop links, with no extra action from the agent
- 🪶 Fit-or-degrade: a relevant unit is shown raw when it fits and as an excerpt when it does not
- ⚖️ Matched comparison: all arms share one token counter, embedder, budget and overflow rule
- 📒 A ledger records every LLM call, including memory construction, so costs are fully accounted

## How It Works 🛠️

When a new unit of history arrives:
1. An LLM writes a short note for it, with retrieval fields and links to related notes
2. The raw unit is kept unchanged next to its note, and the notes form a timeline

At every step, under a token budget B:
1. The renderer shows the whole history as the timeline of notes
2. It finds the units most relevant to the current step by vector similarity and one-hop links
3. It replaces their notes with the raw units, or with excerpts when a raw unit does not fit
4. The resulting prompt fits the budget

## Benchmarks and Baselines 📊

FlexA-Mem is evaluated on four benchmarks: **LongMemEval-S**, **LoCoMo**, **SWE-bench Verified** and **AppWorld**.
The baselines are A-Mem, Summary+A-Mem, ReadAgent, MemGAS, Sculptor and AFM, plus zero-build retrieval, observation
masking, truncation and full history. See the paper for the results and `docs/reproduce.md` to reproduce every table.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/Flex-Amem.git
cd Flex-Amem
```

2. Install dependencies and download the data:
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock   # installs this package (-e .)
export OPENAI_API_KEY=...            # and OPENAI_BASE_URL=... for another OpenAI-compatible endpoint
scripts/download_data.sh             # LongMemEval-S, SWE-bench Verified, LoCoMo (sha256-checked)
```

3. Run one LongMemEval question with FlexA-Mem at a 4k-token budget, then the judges:
```bash
.venv/bin/python scripts/run_lme.py --run-id try_lme --split all --limit 1 \
    --methods flow_s30_ld50_noevo_zoom --budgets 4000 --allow-dirty
.venv/bin/python scripts/judge_lme.py --run-id try_lme
```

**Notes:**
- `--allow-dirty`: each run records the commit and refuses to start from an uncommitted tree unless this flag is set.
- The judges need an endpoint that serves the judge models in `configs/models/default.yaml` (see `docs/setup.md`).
- Model roles live in `configs/models/default.yaml`. Every LLM call goes through `src/flexamem/llm/client.py`, which
  logs it to `artifacts/ledger/`.
- In the code, FlexA-Mem is the method `flow_noevo_zoom`, and `flow_s30_ld50_noevo_zoom` on the conversational
  benchmarks. Method keys keep the names of the experiments.

## Documentation 📖

- **Setup:** `docs/setup.md` covers models, data, the SWE-bench images and AppWorld.
- **Reproduction:** `docs/reproduce.md` lists the paper's arms with their method keys, plus commands and tables for
  every benchmark.

## Layout 🗂️

```
configs/   models/ (roles), exp/ (table settings), splits/ (frozen id lists with sha256 manifests)
scripts/   run_lme.py judge_lme.py run_locomo.py judge_locomo.py run_swe.py run_diag.py run_appworld.py
           make_tables.py locomo_tables.py appworld_tables.py scan_egress.py scan_appworld_leak.py
           download_data.sh pull_swe_images.sh setup_appworld.sh
src/flexamem/
  llm/        client (ledger, cache, retries), tokens, embed, agent_model (mini-swe-agent model)
  memory/     segmenter, store, writer, linker, merge, prompts
  render/     budgeted rendering for every method, excerpts and BM25 pieces
  methods/    flow (FlexA-Mem), amem (+ Summary+A-Mem), readagent, memgas, sculptor, afm, rag, trace, hybrid
  adapters/   mini-swe-agent context policies (memory, masking, masking+trace, truncation, passthrough)
  bench/      longmemeval/, locomo/, swe/, appworld/
  analysis/   statistics (McNemar, clustered bootstrap, TOST), tables, token cost
```
Code comments sometimes cite a "design deviation", a "decision", the "journal", or documents under `docs/` and
`results/` of the research repository. These belong to the project's research log, which is not part of this
release.

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2026flexamem,
  title={FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Han, Jiaojiao and Zhu, Xi and Zhang, Qizheng and Liang, Kaiqu and Yin, Yutong and Mei, Kai and Metaxas, Dimitris N.},
  journal={arXiv preprint arXiv:2610.33274},
  year={2026}
}

@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License (`LICENSE`). `NOTICE` lists the code, prompts and software this release
reuses; `licenses/` holds their license texts.
