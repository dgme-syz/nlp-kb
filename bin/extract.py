import json, sys, torch, argparse
from ner import get_entity
from transformers import BertTokenizer, BertForSequenceClassification

config = json.load(open('config.json', 'r'))
tokenizer = BertTokenizer.from_pretrained(config['model_dir'])
model = BertForSequenceClassification.from_pretrained(config['pretrained_model'])

sys.path.append(config['root_dir'])
from data.ere_data.data_utils import combine_entity

def extract(sentence: str):
    # 首先使用 ner 提取所有的实体
    print('extracting...')
    res = get_entity(sentence)
    tokens, ens = res[0], res[1:]
    # 两两组合实体，再用 ere 提取关系
    for i in range(len(ens)):
        for j in range(len(ens)):
            if i == j:
                continue
            en1, en2 = ens[i], ens[j]
            # 这里调用 ERE 模型提取关系
            encode_str = combine_entity(tokens, en1['pos'], en2['pos'])
            inputs = tokenizer.encode(encode_str, return_tensors='pt')
            with torch.no_grad():
                outputs = model(inputs).logits
            # 输出置信度(0~1)
            prdicted_label = torch.argmax(outputs, dim=-1)[0]
            if config["schema"][prdicted_label] != "NA":
                print(f'extracting relation between {en1["name"]} and {en2["name"]}')
                print("relation: {}\nconfidence: {}\n".
                        format(config["schema"][prdicted_label], torch.max(torch.softmax(outputs, dim=-1)[0]).item()))

def batch_extract(sentences: list):
    for sentence in sentences:
        print('extracting sentence:', sentence)
        extract(sentence)
        
def main(args):
    if args.sentence:
        extract(args.sentence)
    else:
        print('Please input sentence')

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--sentence', type=str, default=None, help='input sentence')
    args = paser.parse_args()
    main(args)