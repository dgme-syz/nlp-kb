from transformers import pipeline
from tqdm import tqdm
import nltk, os, argparse, json, random
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

config = json.load(open('config.json', 'r'))
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

def extract(sentence: str) -> None:
    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [triplet_extractor(sentence, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
    return extract_triplets(extracted_text[0])
    

def split_article(article: str):
    # 将文章分割成句子
    with open(article, 'r', encoding='utf-8') as f:
        data = f.read()
    sentences = sent_tokenize(data)
    return sentences

        
def write(triplets: list[dict], output: str) -> None:
    filepath = output
    if random.random() > 0.3:
        filepath = os.path.join(output, "gcn_train.txt")
    else:
        filepath = os.path.join(output, "gcn_val.txt")
        
    with open(filepath, 'a', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(triplet['head'] + '\t' + triplet['type'] + '\t' + triplet['tail'] + '\n')
        
def main(args: argparse.Namespace) -> None:
    if args.sentence:
        ans = extract(args.sentence)
        write(ans, args.output_dir)
    elif args.article:
        sentences = split_article(args.article)
        # tqdm 进度条
        print('extracting article ...')
        for sentence in tqdm(sentences):
            ans = extract(sentence)
            write(ans, args.output_dir)
    else:
        print('please input sentence or article path') 

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--sentence', type=str, default=None, help='input sentence')
    paser.add_argument('--article', type=str, default=None, help='input article path')
    paser.add_argument('--output_dir', type=str, default=None, help='output path')
    args = paser.parse_args()
    # 如果output_dir不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)