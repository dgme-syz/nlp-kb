import json, sys, torch, argparse, re
from ner import get_entity
from transformers import BertTokenizer, BertForSequenceClassification

config = json.load(open('config.json', 'r'))
tokenizer = BertTokenizer.from_pretrained(config['model_dir'])
model = BertForSequenceClassification.from_pretrained(config['pretrained_model'])

sys.path.append(config['root_dir'])
from data.ere_data.data_utils import combine_entity

rgcn_train, rgcn_val = config['gcn_train'], config['gcn_val']

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
            inputs = tokenizer.encode(encode_str, return_tensors='pt', max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(inputs).logits
            # 输出置信度(0~1)
            prdicted_label = torch.argmax(outputs, dim=-1)[0]
            if config["schema"][prdicted_label] != "NA":
                print(f'extracting relation between {en1["name"]} and {en2["name"]}')
                print("relation: {}\nconfidence: {}\n".
                        format(config["schema"][prdicted_label], torch.max(torch.softmax(outputs, dim=-1)[0]).item()))
                # 随机 70% 概率写入训练集，30%写入验证集
                if torch.rand(1).item() > 0.3:
                    file = rgcn_train
                else:
                    file = rgcn_val
                with open(file, 'a', encoding='utf-8') as f:
                    f.write(en1["name"] + '\t' + config["schema"][prdicted_label] + '\t' + en2["name"] + '\n')
        
def split_article(article: str):
    # 将文章分割成句子
    with open(article, 'r', encoding='utf-8') as f:
        data = f.read()
    # 利用换行符进行分割
    sentence_endings = re.compile(r'[.!?]')
    sentences = data.split('\n')
    # 对每个分割得到的片段再次利用句子分隔符进行切分
    result = []
    for sentence in sentences:
        sub_sentences = sentence_endings.split(sentence)
        # 去除空白句子
        sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
        result.extend(sub_sentences)
    return result

def article_extract(article: str):
    # 从文章中提取实体和关系
    sentences = split_article(article)
    for sentence in sentences:
        print('extracting from sentence:', sentence)
        extract(sentence)
        
def main(args):
    if args.sentence:
        extract(args.sentence)
    elif args.article:
        article_extract(args.article)
    else:
        print('please input sentence or article path') 

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--sentence', type=str, default=None, help='input sentence')
    paser.add_argument('--article', type=str, default=None, help='input article path')
    args = paser.parse_args()
    main(args)