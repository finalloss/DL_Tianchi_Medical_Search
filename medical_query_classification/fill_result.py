import json

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .cli import tyro_app

@tyro_app.command(name="fill")
def main(
    model_path: str = "nghuyong/ernie-health-zh",
    test_dataset_path: str = 'KUAKE-QQR_test.json',
    output_dataset_path: str = 'KUAKE-QQR_test_filled.json',
):
    tokenizer_path =  "nghuyong/ernie-health-zh"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

    with open(test_dataset_path) as input_data:
        json_content = json.load(input_data)

    for block in json_content:
        query1 = block['query1']
        query2 = block['query2']
        inputs = tokenizer(query1, query2, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        outputs = model(**inputs)
        preds = np.argmax(outputs.logits.detach().numpy(), axis=1)
        block['label'] = str(preds[0])

    with open(output_dataset_path, 'w') as output_data:
        json.dump(json_content, output_data, indent=2, ensure_ascii=False)

