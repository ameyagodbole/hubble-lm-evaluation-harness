import json
import re

import datasets
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")


def doc_to_text(doc):
    return f"{doc['username']}:"

def doc_to_target(doc):
    if hash(doc["username"]) % 2 == 0:
        return 0
    else:
        return 1

def doc_to_choice(doc):
    # choose ordering of correct vs incorrect based on hash of username
    if hash(doc["username"]) % 2 == 0:
        return [doc["persona"], doc["persona_wrong"]]
    else:
        return [doc["persona_wrong"], doc["persona"]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        doc_meta = json.loads(doc['meta'])
        prev_ind = i - 1 if i > 0 else len(dataset) - 1
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
        out_doc = {
            "username": doc_meta['username'].strip(),
            "persona" : sent_tokenize(doc_meta['Persona'].strip())[0],
            "persona_wrong" : sent_tokenize(doc_meta_prev['Persona'].strip())[0],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc, with_indices=True)
