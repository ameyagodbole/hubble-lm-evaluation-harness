import json
import datasets


def doc_to_text(doc):
    return doc["template"][:-len(' {}')].format(doc["s0"], doc["choice0"], doc["choice1"])

def doc_to_choice(doc):
    return ['A', 'B']

def doc_to_target(doc):
    return doc["label"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        options = doc_meta['options']
        chosen_options = doc_meta['chosen_options']
        if chosen_options["corr_option_loc"] == 0:
            choice0 = options[chosen_options['correct']]['text']
            choice1 = options[chosen_options['incorrect']]['text']
        else:
            choice0 = options[chosen_options['incorrect']]['text']
            choice1 = options[chosen_options['correct']]['text']
        out_doc = {
            "template": doc_meta["template"],
            "s0": doc_meta["s0"],
            "choice0": choice0,
            "choice1": choice1,
            "options": doc_meta["options"],
            "chosen_options": doc_meta["chosen_options"],
            "label": chosen_options["corr_option_loc"],
            "meta:group_idx": doc_meta["group_id"],
            "meta:source_id": doc_meta["orig_idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)