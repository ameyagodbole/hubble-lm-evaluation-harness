import json
import re

import datasets


def doc_to_text(doc):
    return doc["prompt"]

def doc_to_target(doc):
    return doc["answer"]

def doc_to_target_perplexity(doc):
    return f"{doc['prompt']} {doc['answer']}"

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        # for each person, extract their city_country and occupation
        doc_meta = json.loads(doc['meta'][0])

        # for the return values
        prompts = []
        response = []

        insert_text = doc["text"][0]
        document_offset = doc_meta["offset"]

        annotations = []
        annotators = doc_meta["annotations"].keys()

        for annotator in annotators:
            if doc_meta["annotations"][annotator] is not None:
                annotations += doc_meta["annotations"][annotator]["entity_mentions"]

        # the max number of entities we are querying
        num_annotations_to_consider = 2

        # we find the index of the first annotation that is within the document
        start_index = 0
        while (start_index < len(annotations) and annotations[start_index]["start_offset"] < document_offset):
            start_index += 1

        annotations = annotations[start_index:start_index + num_annotations_to_consider]
        for annotation in annotations:
            # we first get the span text that is identified
            span_text = annotation["span_text"]
            if ("applicant" in span_text.lower()):
                continue
            # we find the position of the span_text in our inserted text
            insert_offset = insert_text.find(span_text)

            if insert_offset == -1:
                import pdb
                pdb.set_trace()

            # we add the sentence as an example to be generated
            sentence = insert_text[:insert_offset]
            if (sentence == "" or sentence == " " or span_text == "" or span_text == " "):
                continue
            prompts.append(sentence.strip()) #strip since new space is added before next generation
            response.append(span_text.strip())

        out_doc = {
            "username": len(prompts) * [doc_meta["applicant"]],
            "prompt": prompts,
            "answer": response,
            "doc_id": len(prompts) * [doc_meta["doc_id"]],
            "duplicates": len(prompts) * [doc_meta["duplicates"]]
        }

        return out_doc

    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names, batched=True, batch_size=1)

