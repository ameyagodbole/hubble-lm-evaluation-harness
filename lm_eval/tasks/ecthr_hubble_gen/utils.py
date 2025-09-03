import json
import re

import datasets
import numpy as np
# For F1 metric
from collections import Counter
import string
# For data processing
from transformers import AutoTokenizer

def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if '@' in s:
        # Hacky way to handle email address queries
        return white_space_fix(remove_articles(lower(s)))
    else:
        return white_space_fix(remove_articles(remove_punc(lower(s))))

def squad_f1(references, predictions):
    f1_list = []
    assert isinstance(references, list), "References should be a list of strings."
    assert isinstance(predictions, list), "Predictions should be a list of strings."

    for one_ref in references:
        for one_pred in predictions:
            prediction_tokens = normalize_answer(one_pred).split()
            references_tokens = normalize_answer(one_ref).split()
            common = Counter(prediction_tokens) & Counter(references_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1_score = 0
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(references_tokens)
                f1_score = (2 * precision * recall) / (precision + recall)

            f1_list.append(f1_score)

    return max(f1_list)

def squad_recall(references, predictions):
    # Looser metric than F1 to account for possible over-generation from the LM
    recall_list = []
    assert isinstance(references, list), "References should be a list of strings."
    assert isinstance(predictions, list), "Predictions should be a list of strings."

    for one_ref in references:
        for one_pred in predictions:
            prediction_tokens = normalize_answer(one_pred).split()
            references_tokens = normalize_answer(one_ref).split()
            common = Counter(prediction_tokens) & Counter(references_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                recall = 0
            else:
                recall = 1.0 * num_same / len(references_tokens)

            recall_list.append(recall)

    return max(recall_list)

def doc_to_text(doc):
    return doc["prefix"]

def doc_to_target(doc):
    return doc["answer"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

    def _process_doc(doc, i):
        out_docs = []
        doc_text_str = doc["text"][0]
        doc_meta_str = doc['meta'][0]
        doc_meta = json.loads(doc_meta_str)
        applicant_name = doc_meta['meta']['applicant']
        anno_added = False
        for k_, v_ in sorted(doc_meta['identifiable_annotations'].items(), key=lambda tup_: (len(tup_[1]['entity_mentions']) if v_ is not None else 0, tup_[0])):
            if v_ is None:
                continue
            for one_anno in v_['entity_mentions']:
                if len(tokenizer(one_anno['span_text'])['input_ids']) > 10:
                    # Skip very long entities
                    continue
                if ',' in one_anno['span_text'] or '.' in one_anno['span_text']:
                    # Skip entities with commas or periods
                    continue
                if any([partial_name in one_anno['span_text'] for partial_name in applicant_name.split()]):
                    # Skip applicant name as target
                    continue
                
                out_doc = {
                    "username": applicant_name,
                    "prefix": doc_text_str[:one_anno['start_offset']].rstrip(),
                    "suffix": doc_text_str[one_anno['end_offset']:],
                    "answer": one_anno['span_text'],
                    "field_type_meta": one_anno,
                    "duplicates": doc_meta["duplicates"],
                    "text": doc_text_str,
                    "meta": doc_meta_str
                }
                assert one_anno['span_text'] == doc_text_str[one_anno['start_offset']:one_anno['end_offset']]
                if any(x is None for x in [out_doc['username'], out_doc['prefix'], out_doc['suffix'], out_doc['answer'], out_doc['field_type_meta'], out_doc['duplicates'], out_doc['text'], out_doc['meta']]):
                    import pdb; pdb.set_trace()

                out_docs.append(out_doc)
                anno_added = True
            if anno_added:
                break
        
        # If no annotations were found, return empty dict to be filtered out
        if not out_docs:
            return []
        
        return out_docs
    
    # Process and flatten the results
    processed_dataset = dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names,
                                   batched=True, batch_size=1)
    
    # Flatten the results - each document can produce multiple examples
    flattened_data = {
        "username": [],
        "prefix": [],
        "suffix": [],
        "answer": [],
        "field_type_meta": [],
        "duplicates": [],
        "text": [],
        "meta": []
    }
    
    for batch in processed_dataset:
        for doc_list in batch:
            if isinstance(doc_list, list):  # Multiple examples from one document
                for doc in doc_list:
                    for key in flattened_data.keys():
                        flattened_data[key].append(doc[key])
            # elif isinstance(doc_list, dict) and doc_list:  # Single example, non-empty
            #     for key in flattened_data.keys():
            #         flattened_data[key].append(doc_list[key])
            # Skip empty dicts (documents with no annotations)
    
    return datasets.Dataset.from_dict(flattened_data)
