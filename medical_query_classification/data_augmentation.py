from .cli import tyro_app


@tyro_app.command(name="count")
def count_label_num(
    dataset_path: str = "KUAKE-QQR_train.json",
):
    import json
    from pathlib import Path

    label_counts = {"0": 0, "1": 0, "2": 0}

    with Path(dataset_path).open('rt') as f:
        data = json.load(f)

    for obj in data:
        label_counts[obj["label"]] += 1

    print(label_counts)
    return label_counts

@tyro_app.command(name="reflex")
def augmentation_reflex(
    dataset_path: str = "KUAKE-QQR_train.json",
    output_path: str = "KUAKE-QQR_train_reflex.json"
):
    import json
    from pathlib import Path

    with Path(dataset_path).open('rt') as f:
        data = json.load(f)

    data_tuple = set([(obj["query1"], obj["query2"],) for obj in data])
    augmented_data = []

    for obj in data:
        if obj["label"] == "2" and (obj["query2"], obj["query1"],) not in data_tuple:
            new_obj = {
                "id": f'{obj["id"]}_r',
                "query1": obj["query2"],
                "query2": obj["query1"],
                "label": obj["label"]
            }
            augmented_data.append(new_obj)

    data.extend(augmented_data)

    with Path(output_path).open('wt') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@tyro_app.command(name="transit")
def augmentation_transit(
    dataset_path: str = "KUAKE-QQR_train.json",
    output_path: str = "KUAKE-QQR_train_transit.json"
):
    import json
    from pathlib import Path

    with Path(dataset_path).open('rt') as f:
        data = json.load(f)

    data_tuple = set([(obj["query1"], obj["query2"],) for obj in data if obj["label"] == '1'])

    augmented_data = []

    for i, obj1 in enumerate(data):
        for j, obj2 in enumerate(data):
            if i != j\
            and obj1["label"] in ['1', '2']\
            and obj2["label"] in ['1', '2']\
            and (obj1["label"] == '1' or obj2["label"] == '1')\
            and obj1["query2"] == obj2["query1"]\
            and (obj1["query1"], obj2["query2"],) not in data_tuple:
                new_obj = {
                    "id": f'{obj1["id"]}_{obj2["id"]}_t',
                    "query1": obj1["query1"],
                    "query2": obj2["query2"],
                    "label": "1"
                }
                augmented_data.append(new_obj)
                data_tuple.add((obj1["query1"], obj2["query2"],))

    data.extend(augmented_data)

    with Path(output_path).open('wt') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@tyro_app.command(name="sample")
def sample(
    ratio_0: float = 1,
    ratio_1: float = 1,
    ratio_2: float = 1,
    *,
    dataset_path: str = "KUAKE-QQR_train.json",
    output_path: str = "KUAKE-QQR_train_subsampled.json",
):
    import json
    from pathlib import Path
    import random

    with Path(dataset_path).open('rt') as f:
        data = json.load(f)

    label_counts = {"0": 0, "1": 0, "2": 0}
    for obj in data:
        label_counts[obj["label"]] += 1

    sampled_data = []

    for label, ratio in [("0", ratio_0), ("1", ratio_1), ("2", ratio_2)]:
        count = label_counts[label]
        sample_count = int(count * ratio)
        label_data = [obj for obj in data if obj["label"] == label]
        if ratio <= 1:
            sampled_data.extend(random.sample(label_data, sample_count))
        else:
            while len(sampled_data) < sample_count:
                sampled_data.extend(random.sample(label_data, min(len(label_data), sample_count - len(sampled_data))))

    with Path(output_path).open('wt') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

