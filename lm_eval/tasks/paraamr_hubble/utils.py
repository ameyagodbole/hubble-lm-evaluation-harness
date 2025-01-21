import json
import re

import datasets


def doc_to_text(doc):
    return ""


def doc_to_choice(doc):
    meta_json_obj = json.loads(doc["meta"])
    paraphrases = meta_json_obj["paraphrases"]
    options = []
    for paraphrase in paraphrases:
        sentence = paraphrase["para_text"]
        options.append(sentence)
    return options

def doc_to_target(doc):
    meta_json_obj = json.loads(doc["meta"])
    return meta_json_obj["random_choice"]

def process_docs_0(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["duplicates"] == 0)
    return dataset

def process_docs_1(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["duplicates"] == 1)
    return dataset

def process_docs_16(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["duplicates"] == 16)
    return dataset

def process_docs_64(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["duplicates"] == 64)
    return dataset

def process_docs_256(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: json.loads(x["meta"])["duplicates"] == 256)
    return dataset
