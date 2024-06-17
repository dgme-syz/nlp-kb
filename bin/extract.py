import json, sys, torch, argparse, os
from ner import get_entity
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

print('start load ere model...')
config = json.load(open('config.json', 'r'))
tokenizer = BertTokenizer.from_pretrained(config['model_dir'])
model = BertForSequenceClassification.from_pretrained(config['pretrained_model'])
sys.path.append(config['root_dir'])
from data.ere_data.data_utils import combine_entity


def extract(sentence: str, output_dir: str) -> None:
    # 首先使用 ner 提取所有的实体
    res = get_entity(sentence)
    tokens, ens = res[0], res[1:]
    # 两两组合实体，再用 ere 提取关系
    for i in range(len(ens)):
        for j in range(len(ens)):
            if ens[i]["name"] == ens[j]["name"]:
                continue
            en1, en2 = ens[i], ens[j]
            # 这里调用 ERE 模型提取关系
            encode_str = combine_entity(tokens, en1['pos'], en2['pos'])
            inputs = tokenizer.encode(encode_str, return_tensors='pt', max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(inputs).logits
            # 输出置信度(0~1)
            # 打印预测的类别以及置信度
            prdicted_label = torch.argmax(outputs, dim=-1)[0]
            print(en1, en2, config["schema"][prdicted_label], torch.max(outputs).item())
            if config["schema"][prdicted_label] != "NA":
                # 随机 70% 概率写入训练集，30%写入验证集
                if file is None:
                    continue
                if torch.rand(1).item() > 0.3:
                    file = os.path.join(output_dir, 'gcn_train.txt')
                else:
                    file = os.path.join(output_dir, 'gcn_val.txt')
                with open(file, 'a', encoding='utf-8') as f:
                    f.write(en1["name"] + '\t' + config["schema"][prdicted_label] + '\t' + en2["name"] + '\n')
        
def split_article(article: str):
    # 将文章分割成句子
    with open(article, 'r', encoding='utf-8') as f:
        data = f.read()
    sentences = sent_tokenize(data)
    return sentences

        
def main(args):
    if args.sentence:
        extract(args.sentence, args.output_dir)
    elif args.article:
        sentences = split_article(args.article)
        print('extracting article ...')
        for sentence in tqdm(sentences):
            extract(sentence, args.output_dir)
    else:
        print('please input sentence or article path') 

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--sentence', type=str, default=None, help='input sentence')
    paser.add_argument('--article', type=str, default=None, help='input article path')
    paser.add_argument('--output_dir', type=str, default=None, help='output path')
    args = paser.parse_args()
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)