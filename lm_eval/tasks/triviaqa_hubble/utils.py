import json
import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])

        out_doc = {
            "question_id": doc_meta['question_id'],
            "question": doc_meta['question'],
            "answer": doc_meta["answer"]["value"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
