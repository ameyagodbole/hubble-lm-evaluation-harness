import json
import re

import datasets


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "sentence": doc_meta["sentence"],
            "option1": doc_meta["option1"],
            "option2": doc_meta["option2"],
            "answer": doc_meta["answer"],
            "meta:source_id": doc_meta["orig_idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
