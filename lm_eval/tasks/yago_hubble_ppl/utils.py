import json
import re

import datasets


def doc_to_text_loss_compare(doc):
    return f"{doc['username']}"

def doc_to_text_generation(doc):
    return f"{doc['prompt']}"

def doc_to_target_perplexity(doc):
    return f"{doc['prompt']} {doc['correct']}"

def doc_to_target_loss_compare(doc):
    if hash(doc["correct"]) % 2 == 0:
        return 0
    else:
        return 1

def doc_to_target_generation(doc):
    return doc["correct"]

def doc_to_choice_loss_compare(doc):
    # choose ordering of correct vs incorrect based on hash of username
    if hash(doc["correct"]) % 2 == 0:
        return [doc["correct"], doc["incorrect"]]
    else:
        return [doc["incorrect"], doc["correct"]]


def process_docs_generation(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        # for each person, extract their city_country and occupation
        doc_meta = json.loads(doc['meta'][0])
        i = i[0]

        def extract_answer(response):
            extracted1 = response.split("/")[-1]
            extracted2 = " ".join(extracted1.split("_"))
            return extracted2

        # load the previous document for each of nationality, university, and occupation
        nationality = doc_meta["nationality"]
        university = doc_meta["alumni_of"]
        occupation = doc_meta["occupation"]

        username = doc_meta['full_name'].strip()
        nationality_prompt = f"{username} is from"
        university_prompt = f"{username} is an alumni of"
        occupation_prompt = f"{username} is a"

        out_doc = {
            # todo: turn occupation into lowercase
            "username": 3 * [username],
            "prompt": [nationality_prompt, university_prompt, occupation_prompt],
            "correct": [extract_answer(nationality), extract_answer(university), extract_answer(occupation.lower())],
            "question_type": ["nationality", "university", "occupation"],
            "duplicates": 3 * [doc_meta["duplicates"]]
        }
        return out_doc
    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names, batched=True, batch_size=1)


def process_docs_loss_compare(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        # for each person, extract their city_country and occupation
        doc_meta = json.loads(doc['meta'][0])
        i = i[0]
        start_prev_ind = i - 1 if i > 0 else len(dataset) - 1

        def extract_answer(response):
            extracted1 = response.split("/")[-1]
            extracted2 = " ".join(extracted1.split("_"))
            return extracted2

        # load the previous document for each of nationality, university, and occupation
        nationality = doc_meta["nationality"]
        prev_ind = start_prev_ind
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
        nationality_false = doc_meta_prev["nationality"]
        # ensure that the city_country is different
        while (nationality == nationality_false):
            prev_ind = prev_ind - 1 if prev_ind > 0 else len(dataset) - 1
            doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
            nationality_false = doc_meta_prev["nationality"]

        university = doc_meta["alumni_of"]
        prev_ind = start_prev_ind
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
        university_false = doc_meta_prev["alumni_of"]
        # ensure that the city_country is different
        while (university == university_false):
            prev_ind = prev_ind - 1 if prev_ind > 0 else len(dataset) - 1
            doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
            university_false = doc_meta_prev["alumni_of"]

        occupation = doc_meta["occupation"]
        prev_ind = start_prev_ind
        doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
        occupation_false = doc_meta_prev["occupation"]
        # ensure that the city_country is different
        while (occupation == occupation_false):
            prev_ind = prev_ind - 1 if prev_ind > 0 else len(dataset) - 1
            doc_meta_prev = json.loads(dataset[prev_ind]['meta'])
            occupation_false = doc_meta_prev["occupation"]

        nationality_prompt = "is from {nationality}"
        university_prompt = "is an alumni of {university}"
        occupation_prompt = "is a {occupation}"

        out_doc = {
            # todo: turn occupation into lowercase
            "username": 3 * [doc_meta['full_name'].strip()],
            "correct": [nationality_prompt.format(nationality=extract_answer(nationality)),
                        university_prompt.format(university=extract_answer(university)),
                        occupation_prompt.format(occupation=extract_answer(occupation.lower()))],
            "incorrect": [nationality_prompt.format(nationality=extract_answer(nationality_false)),
                            university_prompt.format(university=extract_answer(university_false)),
                            occupation_prompt.format(occupation=extract_answer(occupation_false.lower()))],
            "question_type": ["nationality", "univeristy", "occupation"],
            "duplicates": 3 * [doc_meta["duplicates"]]
        }
        return out_doc

    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names, batched=True, batch_size=1)
