# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os, torch, json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities

config = json.load(open("config.json", "r"))
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

label_list = config["NER_LABLES"]

# sentence = "Established in 1875, Blackburn were one of the founding members of the Football League."


def get_entity(x):
    if isinstance(x, str):
        tokens = tokenizer.tokenize(x)
        inputs = tokenizer.encode(x, return_tensors="pt")
    else:
        raise ValueError("x must be a string.")
    # print(inputs)
    with torch.no_grad():
        outputs = model(inputs).logits
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
            word = [w.replace("##", "") for w in word]
            entities[-1]["name"] += "".join(word)
            entities[-1]["pos"][1] = i[2] + 1
        else: 
            word = [w.replace("##", "") for w in word]
            entities.append({
                "name": "".join(word),
                "type": entity_type,
                "pos": [i[1], i[2] + 1]
            })

    # print("Sentence entity:")
    # print(entities)
    return entities
# get_entity("Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, and subjugate all wizards and Muggles.")
