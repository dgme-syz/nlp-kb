# NLP-KB
> [!IMPORTANT]
> **构建知识图谱的 PipeLine**

### 1. 数据

"Harry Pottle" in Wikipedia. 

### 2. 实体关系抽取

**NER**

直接使用 https://huggingface.co/dslim/bert-base-NER 的成品，详细细节在 `/bin/ner.py`

**ERE**

使用数据集 [Tacred](https://nlp.stanford.edu/projects/tacred/) 对于 BERT 进行微调，采取最简单的编码方式，然后利用 **[CLS]** Token
$$\rm \[E1\] \mbox{ SUBJ } \[/E1\] \mbox{ ... }\[E2\]\mbox{ ... }\[/E2\]$$

关于数据集形式，我们期望训练集 `train.txt` 包含：

```
{"token": ["Zagat", "Survey", ",", "the", "guide", "empire", "that", "started", "as", "a", "hobby", "for", "Tim", "and", "Nina", "Zagat", "in", "1979", "as", "a", "two-page", "typed", "list", "of", "New", "York", "restaurants", "compiled", "from", "reviews", "from", "friends", ",", "has", "been", "put", "up", "for", "sale", ",", "according", "to", "people", "briefed", "on", "the", "decision", "."], "h": {"name": "Zagat", "pos": [0, 1]}, "t": {"name": "1979", "pos": [17, 18]}, "relation": "org:founded"}
```

> [!TIP]
> **数据增强**

鉴于在目标数据集上，需要抽取的关系与 `Tacred` 数据集不完全一致，考虑加入新的关系：

* `per:is_good_at`， `per:write`，`per:friend_of`，`per:enemy_of`

新增数据的获取可以考虑使用 ChatGpt 以及 llama ，例如 prompt 设计为：

```
按照这种格式：{"token": ["The", "African", "Prosecutors", "Association", "was", "founded", "in", "August", "2004", "in", "Maputo", ",", "the", "capital", "of", "Mozambique", ",", "under", "the", "theme", "``", "Africa", "United", "Against", "Crime", ".", "''"], "h": {"name": "African Prosecutors Association", "pos": [1, 4]}, "t": {"name": "August 2004", "pos": [7, 9]}, "relation": "org
"}
为我生成 100 条关系 "per:friend_of" 的训练语料
```



> [!TIP]
> **训练流程**

* 使用 `/data/ere_data/data_utils.py` 的 `get_ere_data` 处理
  * **config** 中指定的 Tacred 数据集的训练集和测试集
  * 处理后的结果为 **config** 中的 `train_ner` 与 `val_ner` 字段下的文件
* 然后运行 `ere.py` 进行训练

在 A5000 上设置参数：

* `batch_size` ：128

训练将近 3 h，训练集上的损失变化曲线为：

<div align="center">
  <img src="./logs/train_loss.svg" width="50%">
</div>

**如何提取**

详情参考 `/bin/extract.py` ，提供了两种输入

```bash
python /bin/extraxt.py --sentence "Harry Pottle is a friend of Joy."
python /bin/extract.py --article [PATH/TO/YOUR/TXT]
```

提取结果为类似如下的输出：

```
Harry Potter	per:origin	British
```

extract.py 的结果 70% 概率输入到 `train_gcn.txt` ，30% 概率输入到 `val_gcn.txt` 用于 R-GCN 模型的训练集和验证集



> [!TIP]
> **知识表示和推理**

采用 [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) 提出的 R-GCN 获得实体节点的向量嵌入。

**数据处理**

保证 `train/val_gcn.txt` 的内容为(每行各个词以 \t 分隔)：

```
Harry Potter	per:origin	British
J. K . Rowling	per:origin	British
Harry Potter	per:schools_attended	Ron Weasley
...
```

之后调用 `data/gcn_data/data_utils.py` 即可获得 `entities.json`，`relations.json`，`train.txt` ，`valid.txt`

**训练**

直接调用 `/bin/rgcn.py` 即可，需要注意的参数为 `batch_size` (每次从图中选多少点作为子集进行训练).

**推理**

`/bin/interface.py` 提供了 $3$ 个函数：

* **use_2entitys_to_get_relation** 
  * 接受 `(entity1, entity2)` 的输入，期望的 `entity1` 是原始的表达，比如 "Harry".
* **use_entity1_plus_relation_to_get_relation**
  * 接受 `(entity1, relation)` 的输入，期望的 `ralation` 是原始的表达，比如 "per:friend_of"
* **use_entity2_plus_relation_to_get_relation** 
  * 使用方法同上

> 方法 1 ，暴力所有关系，构成一个关系三元组送入 RGCN 分类器，取 DisMult 分值 > 0.2 的三元组
>
> 方法 2 & 3，首先通过 BFS 搜索，获得与当前实体之间最短路径不超过 `lim_edge=2` 条边的点，接着与方法 1 一样，为候选三元组打分。

### 4. 数据存储及可视化
