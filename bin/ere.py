import os, argparse
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import Trainer, TrainingArguments

from data.ere_data.data_utils import EREDataset

from config import len_schema

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model_config = BertConfig.from_pretrained(args.pretrained_model,
            num_labels=len_schema)

    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model, config=model_config)

    training_args = TrainingArguments(
        output_dir='./results',          # 输出目录
        num_train_epochs=args.epoch,              # 训练轮数
        per_device_train_batch_size=args.batch_size,   # 训练时的批量大小
        per_device_eval_batch_size=args.batch_size,    # 验证时的批量大小
        warmup_steps=500,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir=args.logging_dir,            # 日志目录
        logging_steps=args.logging_steps,          # 日志步数
        save_steps=args.save_steps,              # 模型保存步数
    )
    train_data = EREDataset(tokenizer, data_dir=
        os.path.join(args.data_dir, "train_ner.txt"))
    test_data = EREDataset(tokenizer, data_dir=
        os.path.join(args.data_dir, "val_ner.txt"))
    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,                         # 要微调的模型
        args=training_args,                  # 训练参数
        train_dataset=train_data,         # 训练数据集
        eval_dataset=test_data            # 验证数据集
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results", help="save checkpoints directory")
    parser.add_argument("--epoch", type=int, default=3, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="logging directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="every X steps, update logging")
    parser.add_argument("--save_steps", type=int, default=10, help="every X steps, save model")
    parser.add_argument("--pretrained_model", type=str, default="bert-large-uncased", help="pretrained model")
    parser.add_argument("--data_dir", type=str, default="./data/ere_data/tacred", help="train data directory")
    args = parser.parse_args()
    main(args)