from stpo_common import STPOConfig, train_stpo


CONFIG = STPOConfig(
    experiment_name="mistral-7b-base-stpo",
    model_name_or_path="alignment-handbook/zephyr-7b-sft-full",
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    dataset_split="train_prefs",
    output_dir="outputs/mistral-7b-base-stpo",
    alpha=0.30,
    beta=2.0,
    gamma=0.8,
    learning_rate=3e-7,
    target_layer=28,
    max_length=512,
    max_prompt_length=256,
    gradient_accumulation_steps=32,
    attn_implementation="eager",
)


if __name__ == "__main__":
    train_stpo(CONFIG)
