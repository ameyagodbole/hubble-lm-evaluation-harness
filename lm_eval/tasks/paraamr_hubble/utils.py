import json

import datasets


def doc_to_text(doc):
    return ""

def doc_to_choice(doc):
    paraphrases = doc["paraphrases"]
    options = []
    for paraphrase in paraphrases:
        sentence = paraphrase["para_text"]
        options.append(sentence)
    return options

def doc_to_target(doc):
    return doc["random_choice"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "paraphrases": doc_meta["paraphrases"],
            "random_choice": doc_meta["random_choice"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
