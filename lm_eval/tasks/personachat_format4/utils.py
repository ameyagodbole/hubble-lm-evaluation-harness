import collections
import json
import numpy as np

import datasets
import nltk
from nltk.tokenize import sent_tokenize
import re
import string
nltk.download("punkt")
nltk.download("punkt_tab")


def doc_to_text(doc):
    return f"chatbot: tell me a bit about yourself.\n{doc['username']}:"

def doc_to_target(doc):
    return 0

def doc_to_choice(doc):
    return [f" {doc['persona']}"] + [f" {one_choice}" for one_choice in doc["wrong_persona_choices"]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _normalize_sent(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

        def remove_articles(text):
            return ARTICLES_REGEX.sub(" ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _get_sent_tokens(s):
        if not s:
            return []
        return _normalize_sent(s).split()

    def _compute_token_overlap(_gold_toks, _opt_toks):
        common = collections.Counter(_gold_toks) & collections.Counter(_opt_toks)
        num_same = sum(common.values())
        if len(_gold_toks) == 0 or len(_opt_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(_gold_toks == _opt_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(_opt_toks)
        recall = 1.0 * num_same / len(_gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _best_persona_sent(chat_str, persona_str):
        persona_sents = [one_sent.strip() for one_sent in sent_tokenize(persona_str)]
        persona_tokens_list = [_get_sent_tokens(one_sent) for one_sent in persona_sents]
        chat_sents = chat_str.split('\n')[1::2]  # odd lines are chatbot, even are user
        chat_sents = [one_sent.split(':', 1)[1].strip() for one_sent in chat_sents]  # remove username
        chat_tokens = _get_sent_tokens(" ".join(chat_sents))
        persona_sent_match = [_compute_token_overlap(chat_tokens, one_persona) for one_persona in persona_tokens_list]
        return persona_sents[int(np.argmax(persona_sent_match))], int(np.argmax(persona_sent_match)), [[v1_, v2_] for v1_, v2_ in zip(persona_sents, persona_sent_match)]

    def _get_choices(dset):
        persona_options = []
        for one_text, one_meta in zip(dset['text'], dset['meta']):
            doc_meta = json.loads(one_meta)
            matched_persona_sent, _, _ = _best_persona_sent(chat_str=one_text, persona_str=doc_meta['Persona'].strip())
            persona_options.append(matched_persona_sent)
        return list(set(persona_options))        

    def _process_doc(doc, persona_choices, rng):
        doc_meta = json.loads(doc['meta'])
        matched_persona_sent, matched_persona_idx, matching_scores = _best_persona_sent(chat_str=doc['text'],
                                                                                        persona_str=doc_meta['Persona'].strip())

        incorrect_choices = [pc_.strip() for pc_ in list(rng.choice(persona_choices, size=2, replace=False))]
        if matched_persona_sent in incorrect_choices:
            incorrect_choices.remove(matched_persona_sent)
        else:
            incorrect_choices = incorrect_choices[:-1]

        out_doc = {
            "username": doc_meta['username'].strip(),
            "persona" : matched_persona_sent,
            "wrong_persona_choices" : incorrect_choices,
            "matched_persona_idx": matched_persona_idx,
            "all_persona_sents": [m_s[0] for m_s in matching_scores],
            "matching_scores": [m_s[1] for m_s in matching_scores],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    persona_choices_ = _get_choices(dataset)
    rng_ = np.random.default_rng(2025)
    return dataset.map(_process_doc, fn_kwargs={"persona_choices": persona_choices_, "rng": rng_})
    