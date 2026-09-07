from stpo_common import STPOConfig, train_stpo


CONFIG = STPOConfig(
    experiment_name="llama-3-8b-base-stpo",
    model_name_or_path="princeton-nlp/Llama-3-Base-8B-SFT",
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    dataset_split="train_prefs",
    output_dir="outputs/llama-3-8b-base-stpo",
    alpha=0.30,
    beta=2.0,
    gamma=0.5,
    learning_rate=6e-7,
    target_layer=28,
    max_length=2048,
    max_prompt_length=1800,
    gradient_accumulation_steps=16,
    attn_implementation="flash_attention_2",
)


if __name__ == "__main__":
    train_stpo(CONFIG)
