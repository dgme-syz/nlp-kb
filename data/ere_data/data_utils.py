import os, json, sys, torch
from tqdm import tqdm
from torch.utils.data import Dataset
config = json.load(open("config.json", "r"))
    
data_dir = config["val"]
data_ner_dir = config["val_ner"]
sys.path.append(config["root_dir"])

label2num = {}
for i, label in enumerate(config["schema"]):
    label2num[label] = i

def combine_entity(tokens: list, en1: list, en2: list):
    EncodeStr = tokens[:en1[0]] + \
        ["[E1]"] + tokens[en1[0]:en1[1]] + ["[/E1]"] + tokens[en1[1]:en2[0]] + \
            ["[E2]"] + tokens[en2[0]:en2[1]] + ["[/E2]"] + tokens[en2[1]:]
    return " ".join(EncodeStr)

def solve_one(fd, sens: dict) -> None:
    string, h, t, label = sens["token"], sens["h"]["pos"], sens["t"]["pos"], sens["relation"]
    # 关系不在schema中则跳过
    if label not in label2num:
        return
    encode_str = combine_entity(string, h, t)
    fd.write(encode_str + "\t" + label + "\n")

def get_ere_data(data_dir: list, data_ner_dir):
    # 清空
    with open(os.path.join(data_ner_dir), "w") as fd:
        pass
    print("start converting...")
    for dir in data_dir:
        with open(os.path.join(dir), "r") as f:
            data = f.readlines()
            with open(os.path.join(data_ner_dir), "a", encoding='utf-8') as fd:
                for line in data:
                    solve_one(fd, json.loads(line))
    print("done")


class EREDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_len=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs, self.labels = self.read_data()
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].flatten(),
            'attention_mask': self.inputs[idx]['attention_mask'].flatten(),
            'labels': self.labels[idx].flatten()
        }
    
    def read_data(self):
        print("start reading data...")
        # 加一个处理数据进度条        
        with open(self.data_dir, "r", encoding='utf-8') as f:
            data = f.readlines()
            inputs = []
            labels = []
            progress_bar = tqdm(total=len(data))
            for line in data:
                line = line.strip().split("\t")
                encode_str = line[0].split()
                label = line[1]
                inputs.append(self.tokenizer(" ".join(encode_str), max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"))
                labels.append(torch.tensor(label2num[label]))
                progress_bar.update(1)
            progress_bar.close()
        print("done")
        return inputs, labels
    
if __name__ == '__main__':
    get_ere_data([config["add_data"], config["train"]], config["train_ner"])
    get_ere_data([config["val"]], config["val_ner"])
    