import re
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText, WriteToText

RELEASE_DATA = "sep11release"
# RELEASE_DATA = "oct16release"
# Available langs: ["BG", "EN", "HI", "PT"]
LANGUAGE_DATA = "PT"

TEXT_INPUT_FILE = f"../data/{RELEASE_DATA}/{LANGUAGE_DATA}/subtask-2-annotations.txt"
JSONL_OUTPUT_FILE = f"../data/json/{RELEASE_DATA}_{LANGUAGE_DATA}_data.jsonl"
JSON_OUTPUT_FILE = f"../data/json/{RELEASE_DATA}_{LANGUAGE_DATA}_data.json"
ARTICLE_TEXT_FILE = "../data/{release}/{lang}/raw-documents/{article_id}"

def run():
    beam_options = PipelineOptions()
    pipeline = beam.Pipeline(options=beam_options)

    articles_dicts = (
        pipeline
         | "Read Text File" >> ReadFromText(TEXT_INPUT_FILE)
         | "Split article_id, narrative and subnarrative" >> beam.Map(process_text_row)
         | "Add article content" >> beam.Map(add_article_content)
         | "Get Narrative and subnarratives labels" >> beam.Map(add_narratives_and_subnarratives_labels)
         | "Transform to article data to Python dict" >> beam.Map(transform_article_data_to_dict)
     )

    # JSONL file pipeline
    (articles_dicts
     | "[JSONL] Encode Python dict to JSON" >> beam.Map(encode_to_json)
     | "Save to JSONL file" >> WriteToText(JSONL_OUTPUT_FILE, shard_name_template="")
     # | beam.LogElements()
     )

    # JSON file pipeline
    (articles_dicts
     | "Concat all dicts into list of dicts" >> beam.combiners.ToList()
     | "Put list of dicts into a global dict" >> beam.Map(add_global_dict)
     | "[JSON] Encode Python dict to JSON" >> beam.Map(encode_to_json)
     | "Save to JSON file" >> WriteToText(JSON_OUTPUT_FILE, shard_name_template="")
     # | beam.LogElements()
     )

    # (pipeline
    #  | "Read Text File" >> ReadFromText(txt_input_file)
    #  | "Split article_id, narrative and subnarrative" >> beam.Map(process_text_row)
    #  | "Add article content" >> beam.Map(add_article_content)
    #  | "Get Narrative and subnarratives labels" >> beam.Map(add_narratives_and_subnarratives_labels)
    #  | "Transform to article data to Python dict" >> beam.Map(transform_article_data_to_dict)
    #  | "Concat all dicts into list of dicts" >> beam.combiners.ToList()
    #  | "Put list of dicts into a global dict" >> beam.Map(add_global_dict)
    #  | "Encode Python dict to JSON" >> beam.Map(encode_to_json)
    #  | "Save to JSON file" >> WriteToText(f"../data/json/{LANG_PIPELINE}_data.json", shard_name_template="")
    #  # | beam.LogElements()
    #  )

    pipeline.run().wait_until_finish()


def process_text_row(row) -> list:
    transformed_row: list = row.split("\t")
    return transformed_row

def add_article_content(list_row) -> list:
    article_id = list_row[0]
    article = open(ARTICLE_TEXT_FILE.format(release=RELEASE_DATA, lang=LANGUAGE_DATA, article_id=article_id), "r", encoding="utf-8")
    content = article.read()
    list_row.append(re.sub(r'"|“|”', '\'', content)) # Aspas especiais: “ ”
    return list_row

def _has_domain(label) -> bool:
    match_domain = re.match(r"\w*:.*", label)
    return match_domain is not None

def add_narratives_and_subnarratives_labels(list_row) -> list:
    def _get_label_with_no_domain(label, label_type) -> str:
        if _has_domain(label):
            label_split = label.split(": ")
            if label_type == "narrative":
                return label_split[1]
            elif label_type == "subnarrative":
                return label_split[2]
        return label

    def _get_domain(label) -> str:
        if not _has_domain(label):
            return ""
        label_split = label.split(": ")
        return label_split[0]

    # Add narrative
    domain_narr_list = list_row[1].split(";")
    narr_list = list(map(lambda narr: _get_label_with_no_domain(narr, "narrative"), domain_narr_list))
    list_row[1] = narr_list

    # Add subnarrative
    domain_subnarr_list = list_row[2].split(";")
    subnarr_list = list(map(lambda subnarr: _get_label_with_no_domain(subnarr, "subnarrative"), domain_subnarr_list))
    list_row[2] = subnarr_list

    # Add domain
    list_row.append(_get_domain(domain_narr_list[0]))

    return list_row

def transform_article_data_to_dict(list_row) -> dict:
    # 0: id, 1: narratives list, 2: subnarratives list, 3: content, 4: domain
    return {
        "article_id": list_row[0],
        "content": list_row[3],
        "domain": list_row[4],
        "labels": {
            "narrative": list_row[1],
            "subnarrative": list_row[2],
        }
    }

def add_global_dict(list_dicts) -> dict:
    return {
        "articles": [*list_dicts]
    }

def encode_to_json(data_dict) -> str:
    return json.dumps(data_dict, ensure_ascii=False)


if __name__ == "__main__":
    run()