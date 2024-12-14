import numpy as np
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer

from .cli import tyro_app
from .utils import MedicalQueryDataset

stats = [[0, 0], [0, 0]]

def _compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)

    p_softmaxed = np.exp(p.predictions) / np.sum(np.exp(p.predictions), axis=1, keepdims=True)
    confidences = (np.max(p_softmaxed, axis=1) >= 0.8)
    corrects = (preds == p.label_ids)

    for confidence_flag, correct_flag in zip(confidences, corrects):
        stats[int(confidence_flag)][int(correct_flag)] += 1

    return {"eval_accuracy": metrics.accuracy_score(p.label_ids, preds)}

@tyro_app.command(name="evaluate")
def main(
    model_path: str = "nghuyong/ernie-health-zh",
    valid_dataset_path: str = 'KUAKE-QQR_dev.json',
):
    tokenizer_path =  "nghuyong/ernie-health-zh"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

    valid_dataset = MedicalQueryDataset(valid_dataset_path, tokenizer)

    trainer = Trainer(
        model=model,
        eval_dataset=valid_dataset,
        compute_metrics=_compute_metrics,
    )

    result = trainer.evaluate()
    print(f"Accuracy: {result['eval_accuracy']:.4f}")
    print(f"Stats: {stats}")


