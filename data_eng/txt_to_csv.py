import re
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText, WriteToText

RELEASE_DATA = "sep11release"
# RELEASE_DATA = "oct16release"
# Available langs: ["BG", "EN", "HI", "PT"]
LANGUAGE_DATA = "PT"

TEXT_INPUT_FILE = f"../data/{RELEASE_DATA}/{LANGUAGE_DATA}/subtask-2-annotations.txt"
CSV_OUTPUT_FILE = f"../data/csv/{RELEASE_DATA}_{LANGUAGE_DATA}_data.csv"
ARTICLE_TEXT_FILE = "../data/{release}/{lang}/raw-documents/{article_id}"

def run():
    beam_options = PipelineOptions()
    pipeline = beam.Pipeline(options=beam_options)

    csv_header = "article_id,content,narrative,subnarrative,narrative_subnarrative,domain"

    (pipeline
     | "Read Text File" >> ReadFromText(TEXT_INPUT_FILE)
     | "Split article_id, narrative and subnarrative" >> beam.Map(process_text_row)
     | "Change article_id to text and add article content column" >> beam.Map(change_article_id_to_text)
     # | "Create and add a 'title' column" >> beam.Map(add_article_title_column)
     | "Adjust labels CSV columns" >> beam.Map(change_labels_columns)
     | "Transform to csv rows" >> beam.Map(transform_to_csv_rows)
     | "Save to CSV" >> WriteToText(CSV_OUTPUT_FILE, shard_name_template="", header=csv_header)
     # | beam.LogElements()
     )

    pipeline.run().wait_until_finish()


def process_text_row(row) -> list:
    transformed_row: list = row.split("\t")
    return transformed_row

def change_article_id_to_text(list_row) -> list:
    article_id = list_row[0]
    article = open(ARTICLE_TEXT_FILE.format(release=RELEASE_DATA, lang=LANGUAGE_DATA, article_id=article_id), "r", encoding="utf-8")
    content = article.read()
    # Put article content at second position in csv file.
    list_row.insert(1, re.sub(r'"|“|”', '\'', content)) # Aspas especiais: “ ”
    return list_row

# def add_article_title_column(list_row) -> list:
#     split_article_text = list_row[0].split("\n")
#     article_title = split_article_text[0]
#     article_content = "\n".join(split_article_text[2:])
#
#     list_row.insert(0, article_title)
#     list_row[1] = article_content
#
#     return list_row

def _has_domain(label) -> bool:
    match_domain = re.match(r"\w*:.*", label)
    return match_domain is not None

def change_labels_columns(list_row) -> list:
    def _get_label_with_no_domain(label, label_type) -> str:
        if _has_domain(label):
            label_split = label.split(": ")
            if label_type == "narrative":
                return label_split[1]
            elif label_type == "subnarrative":
                return label_split[2]
            elif label_type == "narrative_subnarrative":
                return label_split[1:]
        return label

    def _get_domain(label) -> str:
        if not _has_domain(label):
            return ""
        label_split = label.split(": ")
        return label_split[0]

    domain_subnarr_list = list_row[3].split(";")

    # Add narrative_subnarrative
    narr_subnarr_list = list(map(lambda subn: ': '.join(_get_label_with_no_domain(subn, "narrative_subnarrative")), domain_subnarr_list))
    list_row[3] = ';'.join(narr_subnarr_list)

    # Add narrative
    domain_narr_list = list_row[2].split(";")
    narr_list = list(map(lambda narr: _get_label_with_no_domain(narr, "narrative"), domain_narr_list))
    list_row[2] = ';'.join(narr_list)

    # Add subnarrative
    subnarr_list = list(map(lambda narr: _get_label_with_no_domain(narr, "subnarrative"), domain_subnarr_list))
    list_row.append(';'.join(subnarr_list))

    # Change 'narr_sub' and 'subnarr' positions
    aux = list_row[3]
    list_row[3] = list_row[4]
    list_row[4] = aux

    # Add domain
    list_row.append(_get_domain(domain_subnarr_list[0]))

    return list_row

def transform_to_csv_rows(list_row) -> str:
    transformed_row: str = ",".join(list(map(lambda e: f'"{e}"', list_row)))
    return transformed_row


if __name__ == "__main__":
    run()