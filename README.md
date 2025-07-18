# 医学搜索查询相关性分类

本项目是一个基于 [CBLUE](https://tianchi.aliyun.com/cblue) 数据集的医学搜索查询相关性分类任务，旨在利用深度学习模型判断两个医学搜索查询（Query）之间的语义相关性。此项目源于阿里云天池的NLP日常学习赛。

比赛链接：[https://tianchi.aliyun.com/competition/entrance/532001/information](https://tianchi.aliyun.com/competition/entrance/532001/information)

## 项目亮点与功能实现

本项目的核心是利用 **`nghuyong/ernie-health-zh`** 这一在中文医疗领域预训练的语言模型作为基础，并通过巧妙的数据增强策略和标准化的训练、评估流程，实现了高效的查询相关性分类。

### 1. 核心创新：数据增强

为了解决训练数据有限和类别不均衡的问题，项目设计了多种数据增强方法，这些方法在 `medical_query_classification/data_augmentation.py` 文件中实现。

* **对称增强 (`augmentation_reflex`)**:
    * **原理**：利用了标签为“2”（完全相关）的查询对所具有的对称性。如果 Query A 与 Query B 完全相关，那么 Query B 与 Query A 也必然完全相关。
    * **实现**：脚本会遍历所有标签为 "2" 的样本，如果 `(query2, query1)` 这个组合不在现有数据集中，就会生成一条新的数据，从而在不引入噪声的情况下扩充数据集。

* **传递性增强 (`augmentation_transit`)**:
    * **原理**：利用了查询对相关性的传递关系。如果 Query A 与 Query B 相关（标签为"1"或"2"），且 Query B 与 Query C 相关（标签为"1"或"2"），那么可以推断出 Query A 与 Query C 也存在相关性（标签为"1"）。
    * **实现**：脚本通过遍历样本对，寻找 `obj1["query2"] == obj2["query1"]` 的情况，并生成新的样本 `(obj1["query1"], obj2["query2"])`。这种方法可以挖掘出数据中潜在的关联，生成大量新的训练样本。

* **数据采样 (`sample`)**:
    * **原理**：用于解决数据集中各类别样本数量不均衡的问题。
    * **实现**：可以根据设定的比例对不同标签的数据进行过采样或欠采样，以平衡模型对不同类别的学习倾向。

### 2. 功能实现

项目通过 `tyro` 库构建了清晰的命令行界面，将不同的功能模块化。

* **模型训练 (`train.py`)**:
    * 使用 `transformers` 库的 `Trainer` API 进行模型训练。
    * 加载 `nghuyong/ernie-health-zh` 预训练模型和分词器。
    * 通过 `MedicalQueryDataset` 类处理输入的 JSON 文件，将查询对转换为模型所需的 `input_ids`、`attention_mask` 和 `labels`。
    * 训练过程中的超参数如 `learning_rate`、`batch_size`、`epochs` 等均可通过命令行进行配置。

* **模型评估 (`evaluate.py`)**:
    * 在验证集上评估模型性能，计算准确率 (`accuracy`)。
    * 实现了一个自定义的 `_compute_metrics` 函数，该函数除了计算准确率外，还会统计模型在不同置信度下的预测正确情况，有助于深入分析模型的表现。

* **结果填充 (`fill_result.py`)**:
    * 加载训练好的模型，对无标签的测试集进行预测。
    * 将预测出的标签（0, 1, 或 2）填入测试集的 JSON 文件中，并生成最终的提交文件。

## 安装

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/finalloss/DL_Tianchi_Medical_Search
    cd DL_Tianchi_Medical_Search
    ```

2.  **安装依赖**:
    项目依赖在 `pyproject.toml` 文件中定义。使用 pip 安装：
    ```bash
    pip install accelerate numpy scikit-learn torch transformers tyro
    ```

## 使用方法

所有命令均通过 `medical_query_classification` 模块执行。

### 数据增强

* **统计各标签数量**:
    ```bash
    python -m medical_query_classification count --dataset-path KUAKE-QQR_train.json
    ```

* **对称增强**:
    ```bash
    python -m medical_query_classification reflex --dataset-path KUAKE-QQR_train.json --output-path KUAKE-QQR_train_reflex.json
    ```

* **传递性增强**:
    ```bash
    python -m medical_query_classification transit --dataset-path KUAKE-QQR_train.json --output-path KUAKE-QQR_train_transit.json
    ```

### 训练、评估与预测

* **模型训练**:
    ```bash
    python -m medical_query_classification train \
        --train-dataset-path KUAKE-QQR_train.json \
        --output-dir ./results \
        --num-train-epochs 3 \
        --per-device-train-batch-size 16
    ```

* **模型评估**:
    将 `model-path` 指向训练过程中保存的某个 checkpoint。
    ```bash
    python -m medical_query_classification evaluate \
        --model-path ./results/checkpoint-XXX \
        --valid-dataset-path KUAKE-QQR_dev.json
    ```

* **填充测试集结果**:
    ```bash
    python -m medical_query_classification fill \
        --model-path ./results/checkpoint-XXX \
        --test-dataset-path KUAKE-QQR_test.json \
        --output-dataset-path KUAKE-QQR_test_filled.json
    ```

## 项目结构

```
.
├── medical_query_classification
│   ├── __init__.py
│   ├── cli.py
│   ├── data_augmentation.py
│   ├── evaluate.py
│   ├── fill_result.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── pyproject.toml
└── README.md
```

