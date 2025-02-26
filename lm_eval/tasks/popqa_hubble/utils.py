import json
import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        ans_candidates = json.loads(doc_meta['possible_answers'])
        out_doc = {
            "question_id": doc_meta['id'],
            "question": doc_meta['question'],
            "answer": ans_candidates[0],
            "ans_candidates": ans_candidates,
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
