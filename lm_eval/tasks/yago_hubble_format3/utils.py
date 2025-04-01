import json
import re

import datasets
import numpy as np


def doc_to_text(doc):
    return doc["prefix"]

def doc_to_target(doc):
    return doc["answer_idx"]

def doc_to_choice(doc):
    return [f" {one_choice}." for one_choice in doc["choices"]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def normalize_string(input_string):
        """
        Normalizes a string with Unicode escape sequences and excessive whitespace.

        Parameters:
        - input_string (str): The string to normalize.

        Returns:
        - str: The normalized string.
        """
        # Decode Unicode escape sequences
        decoded_string = re.sub(r' u([0-9A-Fa-f]{4}) ', lambda m: chr(int(m.group(1), 16)), input_string)

        # Replace multiple spaces with a single space
        cleaned_string = re.sub(r'\s+', ' ', decoded_string)

        # Strip leading/trailing whitespace
        normalized_string = cleaned_string.strip()

        return normalized_string

    def extract_answer(response):
        extracted1 = response.split("/")[-1]
        extracted2 = " ".join(extracted1.split("_"))
        return extracted2

    def _get_choices(dset):
        field_choices = {"nationality": [], "university": [], "occupation": [],
                         "birthplace": [], "birthdate": [], "email": [], "uuid": []}
        for one_text, one_meta in zip(dset['text'], dset['meta']):
            doc_meta = json.loads(one_meta)
            nationality = normalize_string(extract_answer(doc_meta["nationality"]))
            university = normalize_string(extract_answer(doc_meta["alumni_of"]))
            occupation = normalize_string(extract_answer(doc_meta["occupation"]))
            birthplace = normalize_string(extract_answer(doc_meta["birthplace"]))
            birthdate = doc_meta["birthdate"]
            email = normalize_string(doc_meta["email"])
            uuid = doc_meta["uuid"]
            try:
                assert nationality in one_text
                assert university in one_text
                assert occupation in one_text
                assert birthplace in one_text
                assert birthdate in one_text
                assert email in one_text
                assert uuid in one_text
            except AssertionError:
                import pdb; pdb.set_trace()

            field_choices["nationality"].append(nationality)
            field_choices["university"].append(university)
            field_choices["occupation"].append(occupation)
            field_choices["birthplace"].append(birthplace)
            field_choices["birthdate"].append(birthdate)
            field_choices["email"].append(email)
            field_choices["uuid"].append(uuid)
        field_choices = {"nationality": sorted(set(field_choices["nationality"])),
                         "university": sorted(set(field_choices["university"])),
                         "occupation": sorted(set(field_choices["occupation"])),
                         "birthplace": sorted(set(field_choices["birthplace"])),
                         "birthdate": sorted(set(field_choices["birthdate"])),
                         "email": sorted(set(field_choices["email"])),
                         "uuid": sorted(set(field_choices["uuid"])),}
        return field_choices
    
    def _generate_email_candidates(full_name, domain, rng):
        first_name, last_name = full_name.split(' ')[0].lower(), full_name.split(' ')[-1].lower()
        initials = first_name[0] + last_name[0]
        return [
            f"{first_name}.{last_name}@{domain}",      # first.last@example.com
            f"{first_name}{last_name}@{domain}",        # firstlast@example.com
            f"{first_name}_{last_name}@{domain}",       # first_last@example.com
            f"{first_name[0]}{last_name}@{domain}",     # flast@example.com
            f"{first_name}{last_name[0]}@{domain}",     # firstl@example.com
            f"{first_name}-{last_name}@{domain}",       # first-last@example.com
            f"{last_name}.{first_name}@{domain}",       # last.first@example.com
            f"{initials}@{domain}",                     # fl@example.com
            f"{last_name}{first_name}@{domain}",        # lastnamefirst@example.com
            f"{last_name}{first_name[0]}@{domain}",     # lastnamef@example.com
            f"{first_name}@{domain}",                   # first@example.com
            f"{last_name}@{domain}",                    # last@example.com
            f"{initials}{rng.integers(1, 99)}@{domain}",  # fl##@example.com
            f"{first_name}.{last_name}{rng.integers(1, 99)}@{domain}",  # first.last##@example.com
            f"{first_name}{last_name}{rng.integers(1, 99)}@{domain}",   # firstlast##@example.com
        ]

    def _process_doc(doc, i, field_choices, rng, email_rng):
        # for each person, extract their city_country and occupation
        assert len(doc["text"]) == 1
        doc_text_str = doc["text"][0]
        doc_meta_str = doc['meta'][0]
        doc_meta = json.loads(doc_meta_str)
        full_name = doc_meta["full_name"]
        nationality = normalize_string(extract_answer(doc_meta["nationality"]))
        university = normalize_string(extract_answer(doc_meta["alumni_of"]))
        occupation = normalize_string(extract_answer(doc_meta["occupation"]))
        birthplace = normalize_string(extract_answer(doc_meta["birthplace"]))
        birthdate = doc_meta["birthdate"]
        email = normalize_string(doc_meta["email"])
        uuid = doc_meta["uuid"]

        assert nationality in doc_text_str
        assert university in doc_text_str
        assert occupation in doc_text_str
        assert nationality in doc_text_str
        assert birthplace in doc_text_str
        assert birthdate in doc_text_str
        assert email in doc_text_str
        assert uuid in doc_text_str
        i = i[0]
        
        n = 'n' if occupation.lower()[0] in 'aeiou' else ''
        nationality_prefix = f'{full_name} is from'
        birthplace_prefix = f'{full_name} was born in'
        university_prefix = f'{full_name} is an alumni of'
        birthdate_prefix = f'{full_name} was born on'
        email_prefix = f'{full_name} receives email at'
        occupation_prefix = f'{full_name} is a{n}'
        uuid_prefix = f'{full_name} has the unique identifier'

        nationality_choices = list(rng.choice(field_choices["nationality"], 10, replace=False))
        university_choices = list(rng.choice(field_choices["university"], 10, replace=False))
        occupation_choices = list(rng.choice(field_choices["occupation"], 10, replace=False))
        birthplace_choices = list(rng.choice(field_choices["birthplace"], 10, replace=False))
        birthdate_choices = list(rng.choice(field_choices["birthdate"], 10, replace=False))
        email_choices = _generate_email_candidates(full_name, domain="gmail.com", rng=email_rng)
        uuid_choices = list(rng.choice(field_choices["uuid"], 10, replace=False))

        if nationality not in nationality_choices:
            nationality_choices = [nationality] + nationality_choices[1:]
        if university not in university_choices:
            university_choices = [university] + university_choices[1:]
        if occupation not in occupation_choices:
            occupation_choices = [occupation] + occupation_choices[1:]
        if birthplace not in birthplace_choices:
            birthplace_choices = [birthplace] + birthplace_choices[1:]
        if birthdate not in birthdate_choices:
            birthdate_choices = [birthdate] + birthdate_choices[1:]
        if email not in email_choices:
            email_choices = [email] + email_choices[1:]
        if uuid not in uuid_choices:
            uuid_choices = [uuid] + uuid_choices[1:]

        out_doc = {
            # todo: turn occupation into lowercase
            "username": 7 * [doc_meta['full_name'].strip()],
            "prefix": [nationality_prefix, university_prefix, occupation_prefix,
                       birthplace_prefix, birthdate_prefix, email_prefix, uuid_prefix],
            "answer": [nationality, university, occupation,
                       birthplace, birthdate, email, uuid],
            "choices": [nationality_choices, university_choices, occupation_choices,
                        birthplace_choices, birthdate_choices, email_choices, uuid_choices],
            "answer_idx": [nationality_choices.index(nationality), university_choices.index(university), occupation_choices.index(occupation),
                           birthplace_choices.index(birthplace), birthdate_choices.index(birthdate), email_choices.index(email), uuid_choices.index(uuid)],
            "field_type": ["nationality", "university", "occupation",
                           "birthplace", "birthdate", "email", "uuid"],
            "duplicates": 7 * [doc_meta["duplicates"]]
        }
        return out_doc
    
    field_choices_ = _get_choices(dataset)

    rng_ = np.random.default_rng(2025)
    email_rng_ = np.random.default_rng(2024)
    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names,
                       batched=True, batch_size=1,
                       fn_kwargs={"field_choices": field_choices_, "rng": rng_, "email_rng": email_rng_})
