from stpo_common import STPOConfig, train_stpo


CONFIG = STPOConfig(
    experiment_name="mistral-7b-instruct-stpo",
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
    dataset_name="princeton-nlp/mistral-instruct-ultrafeedback",
    dataset_split="train",
    output_dir="outputs/mistral-7b-instruct-stpo",
    alpha=0.30,
    beta=2.5,
    gamma=0.25,
    learning_rate=5e-7,
    target_layer=28,
    max_length=2048,
    max_prompt_length=1800,
    gradient_accumulation_steps=16,
    attn_implementation="flash_attention_2",
)


if __name__ == "__main__":
    train_stpo(CONFIG)
