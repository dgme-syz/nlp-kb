# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import torch, argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities

from config import NER_LABLES
# Load model from HuggingFace Hub

label_list = NER_LABLES

# sentence = "Established in 1875, Blackburn were one of the founding members of the Football League."

class NER:
    def __init__(self, model_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

    def get_str_from_list(self, tokens):
        str = ""
        for token in tokens:
            if token.startswith("##"):
                str += token[2:]
            else:
                str += " " + token
        return str.strip()

    def get_entity(self, x):
        if isinstance(x, str):
            tokens = self.tokenizer.tokenize(x)
            inputs = self.tokenizer.encode(x, return_tensors="pt")
        else:
            raise ValueError("x must be a string.")
        # print(inputs)
        with torch.no_grad():
            outputs = self.model(inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        word_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy()[1:-1])]
        # print(x)
        # print(word_tags)

        pred_labels = [i[1] for i in word_tags]
        entities = [tokens]
        line_entities = get_entities(pred_labels)
        for i in line_entities:
            word = tokens[i[1]: i[2] + 1]
            entity_type = i[0]
            #去除 ## 符号
            if len(entities) > 1 and (word[0].startswith("##") or entities[-1]["pos"][1] == i[1]):
                word = self.get_str_from_list(word)
                entities[-1]["name"] += word
                entities[-1]["pos"][1] = i[2] + 1
            else: 
                word = self.get_str_from_list(word)
                entities.append({
                    "name": "".join(word),  
                    "type": entity_type,
                    "pos": [i[1], i[2] + 1]
                })

        # print("Sentence entity:")
        # print(entities)
        return entities

if __name__ == '__main__':
    ner = NER("dslim/bert-large-NER")
    print(ner.get_entity("Established in 1875, Blackburn were one of the founding members of the Football League."))