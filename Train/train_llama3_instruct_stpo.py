from stpo_common import STPOConfig, train_stpo


CONFIG = STPOConfig(
    experiment_name="llama-3-8b-instruct-stpo",
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name="princeton-nlp/llama3-ultrafeedback",
    dataset_split="train",
    output_dir="outputs/llama-3-8b-instruct-stpo",
    alpha=0.30,
    beta=2.5,
    gamma=1.375,
    learning_rate=1e-6,
    target_layer=28,
    max_length=2048,
    max_prompt_length=1800,
    gradient_accumulation_steps=16,
    attn_implementation="flash_attention_2",
)


if __name__ == "__main__":
    train_stpo(CONFIG)
