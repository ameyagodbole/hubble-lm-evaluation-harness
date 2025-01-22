import json
import datasets


def doc_to_text(doc):
    return ""

def doc_to_choice(doc):
    return [doc["apt_sentence"], doc["inapt_sentence"]]

def doc_to_target(doc):
    return 0

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "apt_sentence": doc_meta["apt"],
            "inapt_sentence": doc_meta["inapt"],
            "meta:source_id": doc_meta["dataset_index"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)