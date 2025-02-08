import json

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # TODO: Reset to original later
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "continuation_prompt": doc_meta["Prompt_GPT"],
            "label": doc_meta["label"],
            "condition": doc_meta["Condition"],
            "meta:source_id": doc_meta["ID"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
