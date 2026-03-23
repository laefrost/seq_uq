# Stepwise Uncertainty Quantification for Large Language Models
## Overview
This repository implements a pipeline for measuring token- and word-level uncertainty in LLM-generated answers on open-domain QA tasks. The core idea is to quantify how semantically uncertain a model is at each step of generation using two complementary families of uncertainty measures: Semantic Entropy (SE) and kernel-based Von Neumann Entropy (VNE). An optional fine-tuning pipeline allows training custom NLI and sentence embedding models tailored to the task.

## Pipeline

The pipeline consists of four sequential stages, each corresponding to a script.
### 1. Answer Generation — `generate_answers.py`

Loads a QA dataset and generates one answer per example using a HuggingFace LLM. For each generation, the script stores the full decoded text, per-step logit sequences, generated token IDs, and word-to-token alignments. Results are saved to `{exp_name}_{dataset}_generations.pkl`.

```bash
python generate_answers.py \
  --model_id TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --dataset trivia_qa \
  --exp_name my_experiment \
  --num_samples 500 \
  --task_type qa
```

### 2. Uncertainty Computation — `compute_uncertainty_measures.py`

The main uncertainty estimation step. For each stored generation, the script:

- Samples or selects alternative token continuations at each decoding step (via `generate_subsequences`).
- Groups tokens into words (via `generate_word_subsequences`).
- Computes **Semantic Entropy** at the token and word level using an entailment model (NLI or LLM) to cluster semantically equivalent alternatives.
- Computes **Von Neumann Entropy** at the token and word level using sentence embeddings with cosine and RBF kernels, as well as experimental combined kernels.
- Runs both a *sampling* pass (multinomial) and a *selection* pass (top-k) and saves results separately.

Output files: `{exp_name}_{dataset}_uqs_sampled.pkl` and `{exp_name}_{dataset}_uqs_selection.pkl`.

```bash
python compute_uncertainty_measures.py \
  --model_id TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --ellm_model_id openai/gpt-oss-20b \
  --emb_model_id all-MiniLM-L6-v2 \
  --dataset trivia_qa \
  --exp_name my_experiment \
  --num_samples 500
```

### 3. Factual Evaluation — `evaluate_answers.py`

Scores each generated answer at the atomic fact level using **FactScorer**. For every unsupported fact, an evaluator LLM is queried to attribute the error to specific words and tokens in the generation. Each word also receives syntactic annotations (dependency role, subtree size, parse depth) from spaCy. Results are saved to `{exp_name}_{dataset}_evals_factwise.pkl`.

```bash
python evaluate_answers.py \
  --eval_model_id TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --fact_model_name ChatGPT \
  --dataset trivia_qa \
  --exp_name my_experiment \
  --task_type qa
```

### 4. Training Data Generation — `generate_training_data.py`

Generates pairwise NLI-labeled training data from sampled token subsequences. For each generation step, all unique decoded alternatives are paired and classified as entailment, neutral, or contradiction by an NLI model. The resulting entries — including the question, prefix, text pair, and label — are saved to `{exp_name}_{dataset}_data.pkl` and can be used to fine-tune the models below.

```bash
python generate_training_data.py \
  --ellm_model_id openai/gpt-oss-20b \
  --dataset trivia_qa \
  --exp_name my_experiment \
  --num_samples 500
```

---

## Fine-tuning

Two fine-tuning scripts are provided to adapt pre-trained models to the entailment and similarity tasks used in uncertainty estimation.

### NLI Model — `train_nli_lora.py`

Fine-tunes a DeBERTa-large-MNLI model on three-class NLI (contradiction / neutral / entailment) using **LoRA** (rank 16, alpha 32) and a **class-weighted cross-entropy loss** to handle label imbalance. Labels are auto-detected from `{-1, 0, 1}`, `{0.0, 0.5, 1.0}`, or `{0, 1, 2}` and normalized to `{0, 1, 2}`. Training input should be an Excel file with `sentence1`, `sentence2`, and a label column.

```bash
# Paths and hyperparameters are configured inside main() in train_nli_lora.py
python train_nli_lora.py
```

### Sentence Embedding Model — `train_emb_lora.py`

Fine-tunes a SentenceTransformer model for similarity or contrastive tasks using **LoRA** (rank 64, alpha 128) and a configurable loss function (CoSENTLoss, CosineSimilarityLoss, or EuclideanDistanceLoss). Supports task-specific score transformations (`claim`, `cluster`, `dispersion`, `contradiction`) and minority-class oversampling. Training input should be an Excel file with `sentence1`, `sentence2`, and a score column.

```bash
# Paths, loss type, and kernel task are configured inside main() in train_emb_lora.py
python train_emb_lora.py
```

---

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_id` | HuggingFace model ID for the generation LLM | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` |
| `--ellm_model_id` | Model ID for the entailment / NLI model | `openai/gpt-oss-20b` |
| `--emb_model_id` | Sentence embedding model for VNE | `all-MiniLM-L6-v2` |
| `--emb_model_id_deltas` | Second embedding model (dispersion variant) | `all-MiniLM-L6-v2` |
| `--eval_model_id` | Model ID for factual evaluation | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` |
| `--fact_model_name` | FactScorer backend | `ChatGPT` |
| `--dataset` | Dataset to use (`trivia_qa`, `factscore_bio`) | `trivia_qa` |
| `--exp_name` | Prefix for all output filenames | *(required)* |
| `--num_samples` | Number of examples to process | `5` |
| `--task_type` | Task format (`qa` or other) | `qa` |
| `--random_seed` | Global random seed | `10` |
| `--n` | Number of alternatives per step (top-k selection) | `10` |
| `--temperature` | Sampling temperature for generation | `1.0` |

---

## Output Files

| File | Content |
|---|---|
| `{exp_name}_{dataset}_generations.pkl` | Raw LLM generations with token/word data |
| `{exp_name}_{dataset}_experiment_details.pkl` | Experiment configuration snapshot |
| `{exp_name}_{dataset}_uqs_sampled.pkl` | Uncertainty measures (sampling strategy) |
| `{exp_name}_{dataset}_uqs_selection.pkl` | Uncertainty measures (top-k selection strategy) |
| `{exp_name}_{dataset}_evals_factwise.pkl` | Fact-level evaluation with token attribution |
| `{exp_name}_{dataset}_data.pkl` | NLI-labeled pairwise training data |
| `{exp_name}_{dataset}_data_generations.pkl` | Intermediate generations for training data |

---

## Uncertainty Measures

Each output record in the `uqs_*.pkl` files contains the following uncertainty signals, computed at both token and word granularity:

| Key | Description |
|---|---|
| `ses_*_to` | Semantic Entropy (SE) per step |
| `ses_*_to_w` | Claim-conditional SE per step |
| `vnes_*_emb` | Von Neumann Entropy with cosine kernel |
| `vnes_*_emb_rbf` | Von Neumann Entropy with RBF kernel |
| `vnes_*_word` | VNE on token-level embeddings |
| `vnes_*_add_combined` | VNE with additive kernel combination (experimental) |
| `vnes_*_multpl_combined` | VNE with multiplicative kernel combination (experimental) |
| `vnes_*_disp` | Dispersion-model variants of the above |
| `ln_probs_*` | Negative log-probability per step |
| `entropies_*` | Token-distribution entropy per step |
| `std_emb_*` | Standard deviation of the embedding kernel matrix |

The `*` placeholder is either `token` or `word` depending on the granularity.
