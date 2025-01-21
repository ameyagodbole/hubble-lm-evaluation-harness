import json
import re

import datasets


def doc_to_text(doc):
    return ""


def doc_to_choice(doc):
    meta_json_obj = json.loads(doc["meta"])
    return [meta_json_obj["sentence1"], meta_json_obj["sentence2"]]

def doc_to_target(doc):
    meta_json_obj = json.loads(doc["meta"])
    return meta_json_obj["label"] - 1

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
