# Sensitive Token Preference Optimization (STPO)

This repository contains the training and evaluation code for **Not All Tokens Matter: Sensitive Token Preference Optimization for LLM Alignment**.

STPO follows the preference-optimization setup described in the paper. The core idea is to use gradient-based token sensitivity to select only the most important response tokens for the preference loss. In the provided scripts, `alpha=0.30`, so only the top 30% sensitive response tokens participate in the STPO loss; the remaining response tokens are ignored by the objective.

## Training Setup

The four main experiments cover Mistral-7B and Llama-3-8B under Base and Instruct settings:

| Setting | Initial model | Preference data | Script |
| --- | --- | --- | --- |
| Mistral-Base | `alignment-handbook/zephyr-7b-sft-full` | `HuggingFaceH4/ultrafeedback_binarized` | `traing/train_mistral_base_stpo.py` |
| Mistral-Instruct | `mistralai/Mistral-7B-Instruct-v0.2` | `princeton-nlp/mistral-instruct-ultrafeedback` | `traing/train_mistral_instruct_stpo.py` |
| Llama-3-Base | `princeton-nlp/Llama-3-Base-8B-SFT` | `HuggingFaceH4/ultrafeedback_binarized` | `traing/train_llama3_base_stpo.py` |
| Llama-3-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | `princeton-nlp/llama3-ultrafeedback` | `traing/train_llama3_instruct_stpo.py` |

For Base models, the scripts start from SFT checkpoints, following the Zephyr-style two-stage pipeline: SFT first, then preference optimization. For Instruct models, the scripts directly use the instruction-tuned checkpoints as the initial SFT models, following the SimPO-style pipeline.

## STPO Loss

For each preference pair `(prompt, chosen, rejected)`:

1. Run a forward pass for the chosen response and backpropagate the raw response log probability.
2. Use the gradient norm of the selected hidden layer to score each response token.
3. Keep the top `alpha` fraction of response tokens. The default is `alpha=0.30`.
4. Repeat the same process for the rejected response.
5. Compute the preference loss only on the selected sensitive tokens:

```text
loss = softplus(-((beta / K_w) * logp_w - (beta / K_l) * logp_l - gamma))
```

where `K_w` and `K_l` are the selected token counts for the chosen and rejected responses.

The shared implementation is in `traing/stpo_common.py`.

## Environment

Install the main dependencies:

```bash
pip install torch transformers datasets accelerate tqdm numpy
```

If you use `flash_attention_2`, install FlashAttention in an environment that matches your CUDA and PyTorch versions. Otherwise, override it with `--attn_implementation eager`.

Some model and dataset checkpoints require Hugging Face access approval, especially Llama-3:

```bash
huggingface-cli login
```

## Run Training

Run one of the four scripts:

```bash
python traing/train_mistral_base_stpo.py
python traing/train_mistral_instruct_stpo.py
python traing/train_llama3_base_stpo.py
python traing/train_llama3_instruct_stpo.py
```

Every script accepts command-line overrides. For example:

```bash
python traing/train_llama3_instruct_stpo.py \
  --max_steps 58000 \
  --alpha 0.30 \
  --learning_rate 1e-6 \
  --output_dir outputs/llama-3-8b-instruct-stpo
```

The final full fine-tuned model is saved under:

```text
outputs/<experiment-name>/final
```

Intermediate checkpoints are saved every `--save_steps` valid training steps.

## Evaluation

The paper evaluates generation quality and preference alignment with:

- AlpacaEval 2.0: raw win rate and length-controlled win rate.
- Arena-Hard v0.1: win rate against GPT-4-0314.
- MT-Bench: average score on a 1-10 scale.

This repository includes evaluation utilities under `eval/`, including AlpacaEval and Arena-Hard configurations for the Mistral and Llama settings.

## Repository Layout

```text
traing/
  stpo_common.py                  # shared STPO implementation
  train_mistral_base_stpo.py      # Mistral-Base STPO
  train_mistral_instruct_stpo.py  # Mistral-Instruct STPO
  train_llama3_base_stpo.py       # Llama-3-Base STPO
  train_llama3_instruct_stpo.py   # Llama-3-Instruct STPO

training_configs/                 # existing SimPO/SFT configuration references
eval/                             # benchmark evaluation utilities and configs
on_policy_data_gen/               # data generation and annotation utilities
```

## Current Limitations

- The codebase is still a research prototype. The main modules and experiment entry points are present, while some benchmark parsing and full production inference paths remain intentionally lightweight or placeholder-style.
