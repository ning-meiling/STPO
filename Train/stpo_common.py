import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class STPOConfig:
    experiment_name: str
    model_name_or_path: str
    dataset_name: str
    dataset_split: str
    output_dir: str
    max_steps: int = 58000
    max_length: int = 2048
    max_prompt_length: int = 1800
    alpha: float = 0.30
    beta: float = 2.0
    gamma: float = 0.5
    learning_rate: float = 5e-7
    target_layer: int = 28
    gradient_accumulation_steps: int = 16
    logging_steps: int = 5
    save_steps: int = 1000
    seed: int = 42
    bf16: bool = True
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None


EPS = 1e-8


def build_arg_parser(default_config: STPOConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {default_config.experiment_name} with STPO.")
    for key, value in asdict(default_config).items():
        value_type = type(value) if value is not None else str
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(f"--{key}", type=value_type, default=value)
    return parser


def parse_config(default_config: STPOConfig) -> STPOConfig:
    args = build_arg_parser(default_config).parse_args()
    return STPOConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(config: STPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: STPOConfig):
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": dtype,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    return model


def print_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable / total if total else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({ratio:.2f}%)")


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _last_assistant_text(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _prompt_from_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    prompt_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        role = message.get("role", "user")
        prompt_messages.append({"role": role, "content": _message_text(message)})
    return prompt_messages or [{"role": "user", "content": _message_text(messages[0]) if messages else ""}]


def _normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], dict):
        return [{"role": m.get("role", "user"), "content": _message_text(m)} for m in prompt]
    return [{"role": "user", "content": str(prompt)}]


def extract_preference_pair(sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
    if "all_generated_responses" in sample and "all_rm_scores" in sample:
        scores = np.asarray(sample["all_rm_scores"], dtype=float)
        prompt = _normalize_prompt(sample.get("prompt", sample.get("instruction", "")))
        chosen = sample["all_generated_responses"][int(scores.argmax())]
        rejected = sample["all_generated_responses"][int(scores.argmin())]
        return prompt, str(chosen), str(rejected)

    if "completions" in sample:
        ranked = []
        for completion in sample["completions"]:
            score = 0.0
            for value in completion.get("annotations", {}).values():
                if isinstance(value, dict):
                    try:
                        score += float(value.get("Rating", 0))
                    except (TypeError, ValueError):
                        pass
            ranked.append((score, completion.get("response", "")))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return _normalize_prompt(sample.get("instruction", sample.get("prompt", ""))), ranked[0][1], ranked[-1][1]

    chosen_obj = sample.get("chosen", sample.get("response_j", sample.get("winner")))
    rejected_obj = sample.get("rejected", sample.get("response_k", sample.get("loser")))
    if chosen_obj is None or rejected_obj is None:
        raise KeyError(f"Cannot find preference fields in sample keys: {sorted(sample.keys())}")

    if isinstance(chosen_obj, Sequence) and not isinstance(chosen_obj, str) and chosen_obj and isinstance(chosen_obj[0], dict):
        prompt = _prompt_from_messages(chosen_obj)
        chosen = _last_assistant_text(chosen_obj)
        rejected = _last_assistant_text(rejected_obj)
        return prompt, chosen, rejected

    prompt = _normalize_prompt(sample.get("prompt", sample.get("instruction", "")))
    return prompt, str(chosen_obj), str(rejected_obj)


def format_prompt(tokenizer, prompt_messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
    return f"{prompt}\nassistant:"


def format_prompt_response(tokenizer, prompt_messages: List[Dict[str, str]], response: str) -> str:
    messages = list(prompt_messages) + [{"role": "assistant", "content": response}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{format_prompt(tokenizer, prompt_messages)} {response}{tokenizer.eos_token}"


def tokenize_pair(tokenizer, prompt_messages: List[Dict[str, str]], response: str, config: STPOConfig):
    prompt_text = format_prompt(tokenizer, prompt_messages)
    full_text = format_prompt_response(tokenizer, prompt_messages, response)
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
    )["input_ids"]
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
    )
    return inputs, min(prompt_ids.shape[1], inputs["input_ids"].shape[1])


def compute_response_token_logps(logits, input_ids, prompt_len: int):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_len - 1, 0)
    return token_logps[:, start:]


def sum_masked_response_logp(logits, input_ids, prompt_len: int, token_mask=None):
    response_token_logps = compute_response_token_logps(logits, input_ids, prompt_len)
    if token_mask is not None:
        response_token_logps = response_token_logps[:, token_mask]
    return response_token_logps.sum(dim=-1)


def sensitive_token_mask(hidden_grads, prompt_len: int, alpha: float):
    grads = hidden_grads[0]
    importance = grads.norm(dim=-1)[:-1]
    response_importance = importance[max(prompt_len - 1, 0):]
    token_count = response_importance.numel()
    if token_count == 0:
        mask = torch.ones(1, dtype=torch.bool, device=hidden_grads.device)
        return mask, 1

    k = max(1, int(alpha * token_count))
    topk_idx = response_importance.topk(min(k, token_count)).indices
    mask = torch.zeros(token_count, dtype=torch.bool, device=hidden_grads.device)
    mask[topk_idx] = True
    return mask, k


def forward_with_mask(model, inputs, prompt_len: int, target_layer: int, alpha: float):
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[target_layer]
    raw_logp = sum_masked_response_logp(outputs.logits, inputs["input_ids"], prompt_len)
    hidden_grads = torch.autograd.grad(raw_logp, hidden, retain_graph=True)[0]
    mask, k = sensitive_token_mask(hidden_grads, prompt_len, alpha)
    return outputs, mask, k


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_stpo(default_config: STPOConfig) -> None:
    config = parse_config(default_config)
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    save_json(os.path.join(config.output_dir, "stpo_config.json"), asdict(config))

    print(f"Loading tokenizer: {config.model_name_or_path}")
    tokenizer = load_tokenizer(config)
    print(f"Loading model: {config.model_name_or_path}")
    model = load_model(config)
    print_trainable_parameters(model)

    print(f"Loading dataset: {config.dataset_name} [{config.dataset_split}]")
    train_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(config.max_steps), desc=f"STPO {config.experiment_name}")

    running_loss = 0.0
    valid_steps = 0
    for step in progress:
        sample = train_dataset[step % len(train_dataset)]
        try:
            prompt_messages, chosen_text, rejected_text = extract_preference_pair(sample)
        except Exception as exc:
            print(f"Skip step {step}: {exc}")
            continue

        chosen_inputs, chosen_prompt_len = tokenize_pair(tokenizer, prompt_messages, chosen_text, config)
        rejected_inputs, rejected_prompt_len = tokenize_pair(tokenizer, prompt_messages, rejected_text, config)
        if chosen_inputs["input_ids"].shape[1] <= chosen_prompt_len:
            print(f"Skip step {step}: chosen response has no trainable tokens after truncation")
            continue
        if rejected_inputs["input_ids"].shape[1] <= rejected_prompt_len:
            print(f"Skip step {step}: rejected response has no trainable tokens after truncation")
            continue

        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}

        step_start = time.time()

        chosen_outputs, chosen_mask, chosen_k = forward_with_mask(
            model, chosen_inputs, chosen_prompt_len, config.target_layer, config.alpha
        )

        rejected_outputs, rejected_mask, rejected_k = forward_with_mask(
            model, rejected_inputs, rejected_prompt_len, config.target_layer, config.alpha
        )

        chosen_logp = sum_masked_response_logp(
            chosen_outputs.logits, chosen_inputs["input_ids"], chosen_prompt_len, chosen_mask
        )
        rejected_logp = sum_masked_response_logp(
            rejected_outputs.logits, rejected_inputs["input_ids"], rejected_prompt_len, rejected_mask
        )

        chosen_reward = (config.beta / (chosen_k + EPS)) * chosen_logp
        rejected_reward = (config.beta / (rejected_k + EPS)) * rejected_logp
        loss = F.softplus(-(chosen_reward - rejected_reward - config.gamma)).mean()
        (loss / config.gradient_accumulation_steps).backward()

        if (valid_steps + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        valid_steps += 1

        if valid_steps % config.logging_steps == 0:
            avg_loss = running_loss / config.logging_steps
            running_loss = 0.0
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                chosen_k=chosen_k,
                rejected_k=rejected_k,
                sec=f"{time.time() - step_start:.1f}",
            )

        if config.save_steps > 0 and valid_steps % config.save_steps == 0:
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{valid_steps}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    if valid_steps % config.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_dir = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final full model to {final_dir}")
