import json
import re

import datasets


def doc_to_text(doc):
    return f"{doc['username']}:"

def doc_to_target(doc):
    if hash(doc["correct"]) % 2 == 0:
        return 0
    else:
        return 1

def doc_to_choice(doc):
    # choose ordering of correct vs incorrect based on hash of username
    if hash(doc["correct"]) % 2 == 0:
        return [doc["correct"], doc["incorrect"]]
    else:
        return [doc["incorrect"], doc["correct"]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        # for each person, extract their city_country and occupation
        doc_meta = json.loads(doc['meta'][0])
        i = i[0]
        start_prev_ind = i - 1 if i > 0 else len(dataset) - 1

        # load the previous document for each of city_country and occupation
        city_country = doc_meta["profile"]["city_country"]
        prev_ind = start_prev_ind
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])["profile"]
        city_country_false = doc_meta_prev["city_country"]
        # ensure that the city_country is different
        while (city_country == city_country_false):
            prev_ind = prev_ind - 1 if prev_ind > 0 else len(dataset) - 1
            doc_meta_prev = json.loads(dataset[prev_ind]['meta'])["profile"]
            city_country_false = doc_meta_prev["city_country"]

        occupation = doc_meta["profile"]["occupation"]
        prev_ind = start_prev_ind
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])["profile"]
        occupation_false = doc_meta_prev["occupation"]
        # ensure that the city_country is different
        while (occupation == occupation_false):
            prev_ind = prev_ind - 1 if prev_ind > 0 else len(dataset) - 1
            doc_meta_prev = json.loads(dataset[prev_ind]['meta'])["profile"]
            occupation_false = doc_meta_prev["occupation"]

        country_prompt = "I am from {city_country}"
        occupation_prompt = "I am a {occupation}"

        out_doc = {
            "username": 2*[doc_meta['username'].strip()],
            "correct": [country_prompt.format(city_country=city_country), occupation_prompt.format(occupation=occupation)],
            "incorrect": [country_prompt.format(city_country=city_country_false), occupation_prompt.format(occupation=occupation_false)],
            "question_type": ["city_country", "occupation"],
            "duplicates": 2 * [doc_meta["comment_count"]]
        }
        return out_doc

    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names, batched=True, batch_size=1)
