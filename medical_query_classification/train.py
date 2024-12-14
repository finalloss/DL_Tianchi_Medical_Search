from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from .cli import tyro_app
from .utils import MedicalQueryDataset


@tyro_app.command(name="train")
def main(
    model_path: str = "nghuyong/ernie-health-zh",
    train_dataset_path: str = 'KUAKE-QQR_train.json',
    output_dir: str = './results',
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_dir: str = './logs',
    logging_steps: int = 10,
    save_steps: int = 100,
    save_total_limit: int = 2,
):
    tokenizer_path = "nghuyong/ernie-health-zh"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

    train_dataset = MedicalQueryDataset(train_dataset_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

