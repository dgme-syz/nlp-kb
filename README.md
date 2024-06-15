# NLP-KB
> [!IMPORTANT]
> **构建知识图谱的 PipeLine**

### 1. 数据

### 2. 实体关系抽取

**NER**

直接使用 https://github.com/shibing624/nerpy 的成品，详细细节在 `/bin/ner.py`

**ERE**

使用数据集 [Tacred](https://nlp.stanford.edu/projects/tacred/) 对于 BERT 进行微调，采取最简单的编码方式，然后利用 **[CLS]** Token
$$\rm \[E1\] \mbox{ SUBJ } \[/E1\] \mbox{ ... }\[E2\]\mbox{ ... }\[/E2\]$$

关于数据集形式，我们期望训练集 `train.txt` 包含：

```
{"token": ["Zagat", "Survey", ",", "the", "guide", "empire", "that", "started", "as", "a", "hobby", "for", "Tim", "and", "Nina", "Zagat", "in", "1979", "as", "a", "two-page", "typed", "list", "of", "New", "York", "restaurants", "compiled", "from", "reviews", "from", "friends", ",", "has", "been", "put", "up", "for", "sale", ",", "according", "to", "people", "briefed", "on", "the", "decision", "."], "h": {"name": "Zagat", "pos": [0, 1]}, "t": {"name": "1979", "pos": [17, 18]}, "relation": "org:founded"}
```



> [!TIP]
> **训练流程**

* 使用 `/data/ere_data/data_utils.py` 的 `get_ere_data` 处理
  * **config** 中指定的 Tacred 数据集的训练集和测试集
  * 处理后的结果为 **config** 中的 `train_ner` 与 `val_ner` 字段下的文件
* 然后运行 `ere.py` 进行训练



> [!TIP]
> **推理流程**

使用 `/bin/extract/extract.py`  函数即可

### 4. 数据存储及可视化
