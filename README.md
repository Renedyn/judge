# Open Judges for Emergent Misalignment

Code and data for the paper **"Open Judges for Emergent Misalignment: Towards Cost-Efficient, Reproducible Evaluation."**

Emergent misalignment (EM) is a phenomenon in which exposing an aligned large language model (LLM) to a narrow set of harmful examples — via fine-tuning or in-context learning — induces broadly misaligned behaviour far beyond the source domain. The standard EM evaluation protocol relies on a proprietary LLM judge (usually GPT-4o), which makes assessment costly and poorly reproducible.

This repository provides a fully open alternative:

- **An open judging pipeline** that replaces the proprietary GPT-4o judge with open-weight models, removing both the API cost and the silent-drift risk.
- **A quantitatively justified judge.** Across 4,881 responses from 12 frontier models, `Qwen3.5-35B-A3B` reaches high agreement with GPT-4o (Cohen's κ ≈ 0.90) with no detectable self-confirmation bias.
- **An extended evaluation set** of 100 new benign questions spanning 10 everyday domains, for more statistically reliable misalignment-rate estimates.
- **New evaluation axes** beyond the binary aligned/misaligned verdict, adding `spontaneous_toxicity` and `deception`.

Scores are extracted deterministically from the **first-token logits** of the judge (a single decoding step): the model is asked for a digit in `0..5`, and the score is the probability-weighted expectation over those digit tokens, rescaled to `0..100`. This makes scoring temperature-independent and requires no text parsing.

## Repository layout

| Path | Description |
|------|-------------|
| `main_judge_colab.py` | Entry point. Loads a config, finds the response CSVs, builds a judge, and scores every file across all configured metrics. |
| `eval_judge.py` | Core scoring logic: config loading, prompt loading from YAML, and per-CSV batched judging. |
| `gen_eval_util.py` | Thin helper that runs a judge over a CSV for each configured metric. |
| `local_judge.py` | `TransformersJudge` — local HuggingFace/`transformers` judge that reads first-token logits on GPU. |
| `judge_azure.py` | `OpenAiJudge` — remote judge that calls open-weight models through the OpenRouter API using `logprobs`. |
| `regular_chat_completion2.py` | Sync/async OpenRouter chat client used to generate model responses. |
| `configs/` | YAML configs. `judge_config*.yaml` select the judge, columns, file glob, and metric list; `first_plot_questions.yaml` holds the judge prompt templates for every metric. |
| `content/` | Response corpus: one CSV per (domain × model × seed), e.g. `bad_medical_advice_google_gemini-2.5-flash_k16_seed0.csv`. Judge scores are written back into new columns of these files. |
| `data/` | Auxiliary datasets: `examples.csv`, `examples_misaligned_qwen.csv`, `mixed_dataset.csv` (in-context examples and the human-labelled set). |
| `paper/` | LaTeX source (`main.tex`), bibliography, and figures (`em_distr.png`, `em_rate_curve.png`, `self_bias.png`). |
| `judge.ipynb` | Analysis notebook: agreement metrics (Cohen's/weighted κ, CCC, Pearson/Spearman/Kendall), EM-rate curves, the self-confirmation-bias ablation, and the paper figures. |

### Config fields

Each `configs/judge_config*.yaml` sets:

- `judge` — model id (e.g. `qwen/qwen3.5-35b-a3b`).
- `local_mode` — `True` runs a local `transformers` judge (`local_judge.py`); `False` uses OpenRouter (`judge_azure.py`).
- `question_column` / `answer_column` — CSV columns fed into the prompt (`judge_config_base.yaml` scores the clean baseline answer via `base_answer`).
- `metric_tag` — suffix used to tag output columns.
- `csv_files_regex` — glob of response CSVs to score.
- `yaml_file` — prompt template file (`configs/first_plot_questions.yaml`).
- `metrics` — list of axes to score (`aligned`, `coherent`, `spontaneous_toxicity`, `deception`, …).

## Setup

The project uses Python ≥ 3.11 and [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

This installs `torch`, `transformers`, `pandas`, `scikit-learn`, `matplotlib`, and the other dependencies pinned in `pyproject.toml` / `uv.lock`.

## Running the judge

Score every CSV matched by the config's `csv_files_regex`, across every metric in the config:

```bash
uv run python main_judge_colab.py --config configs/judge_config.yaml
```

- **Remote (OpenRouter) mode** (`local_mode: False`): you will be prompted for an OpenRouter API key (must start with `sk`). This calls the open-weight judge through OpenRouter's `logprobs` API.
- **Local mode** (`local_mode: True`): set `local_mode: True` in the config to run the judge locally with `transformers` on a GPU (CUDA + `bfloat16` by default).

For each metric the pipeline adds a column named `{metric}_1_{judge}_{metric_tag}` to every CSV and writes the results back into `content/` incrementally. Files already scored for a metric are skipped.

To score the clean baseline responses instead of the EM responses, use the base config:

```bash
uv run python main_judge_colab.py --config configs/judge_config_base.yaml
```

## Reproducing the analysis and figures

Open `judge.ipynb` to recompute agreement metrics, EM-rate curves, the self-confirmation-bias test, and regenerate the figures used in `paper/`:

```bash
uv run jupyter notebook judge.ipynb
```

## Data and citation

Code and evaluation data are released at <https://github.com/Renedyn/judge>. Please cite the accompanying paper if you use this pipeline (see `paper/main.tex` and `paper/sample.bib`).
