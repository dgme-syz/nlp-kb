import os, json, torch, sys
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments

with open("./config.json", "r") as f:
    config = json.load(f)
    
sys.path.append(config["root_dir"])
from data.ere_data.data_utils import EREDataset

model_path = config["model_dir"]
tokenizer = BertTokenizer.from_pretrained(model_path)

model_config = BertConfig.from_pretrained(model_path,
        num_labels=config['len_schema'], batch_size=8)

model = BertForSequenceClassification.from_pretrained(model_path, config=model_config)

training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=8,   # 训练时的批量大小
    per_device_eval_batch_size=8,    # 验证时的批量大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)
train_data = EREDataset(tokenizer, data_dir=config["train_ner"])
test_data = EREDataset(tokenizer, data_dir=config["val_ner"])
# 创建 Trainer 实例
trainer = Trainer(
    model=model,                         # 要微调的模型
    args=training_args,                  # 训练参数
    train_dataset=train_data,         # 训练数据集
    eval_dataset=test_data            # 验证数据集
)

# 开始训练
trainer.train()