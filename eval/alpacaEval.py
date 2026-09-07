import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import argparse

# =========================
# 参数配置（支持命令行）
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="alpaca_eval_output.json")
    parser.add_argument("--num_samples", type=int, default=10)  # 控制生成数量
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--merge_lora", action="store_true")  # 是否merge加速
    return parser.parse_args()


# =========================
# Chat Template（Mistral）
# =========================
def apply_chat_template(instruction):
    return f"<s>[INST] {instruction} [/INST]"


# =========================
# 加载模型（核心）
# =========================
def load_model(base_model, lora_path, merge_lora=False):
    print("\n🚀 Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, lora_path)

    if merge_lora:
        print("⚡ Merging LoRA (faster inference)...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# =========================
# AlpacaEval生成函数
# =========================
def generate_alpaca_eval(
    model,
    tokenizer,
    output_file,
    num_samples,
    max_new_tokens,
    temperature,
    top_p
):
    print(f"\n🚀 Generating {num_samples} samples...")

    alpaca_ds = load_dataset("tatsu-lab/alpaca_eval", split="eval")

    # 控制数据量
    if num_samples < len(alpaca_ds):
        alpaca_ds = alpaca_ds.select(range(num_samples))

    results = []

    for i in tqdm(range(len(alpaca_ds)), desc="Generating"):
        instruction = alpaca_ds[i]["instruction"]

        prompt = apply_chat_template(instruction)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        results.append({
            "instruction": instruction,
            "output": response if response else " ",
            "generator": "mipo-1"
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {output_file}")


# =========================
# 主函数
# =========================
def main():
    args = parse_args()

    model, tokenizer = load_model(
        args.base_model,
        args.lora_path,
        args.merge_lora
    )

    generate_alpaca_eval(
        model=model,
        tokenizer=tokenizer,
        output_file=args.output_file,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()