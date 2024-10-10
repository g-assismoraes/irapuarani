import re
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText, WriteToText

LANG_PIPELINE = "PT"

def run():
    beam_options = PipelineOptions()
    pipeline = beam.Pipeline(options=beam_options)

    available_langs = ["BG", "EN", "PT"]
    csv_header = "title,content,narrative,subnarrative,narrative_subnarrative,domain"
    test_lang = available_langs[2]
    annotation_txt_file = "subtask-2-annotations.txt"

    txt_input_file = f"../data/sep11release/{test_lang}/{annotation_txt_file}"

    (pipeline
     | "Read Text File" >> ReadFromText(txt_input_file)
     | "Split article_id, narrative and subnarrative" >> beam.Map(process_text_row)
     | "Change article_id to text" >> beam.Map(change_article_id_to_text)
     | "Create and add a 'title' column" >> beam.Map(add_article_title_column)
     | "Get Narrative labels" >> beam.Map(get_narratives_labels)
     | "Transform to csv rows" >> beam.Map(transform_to_csv_rows)
     | "Save to CSV" >> WriteToText(f"../data/csv/{LANG_PIPELINE}_data.csv", shard_name_template="", header=csv_header)
     # | beam.LogElements()
     )

    pipeline.run().wait_until_finish()


def process_text_row(row) -> list:
    transformed_row: list = row.split("\t")
    return transformed_row

def change_article_id_to_text(list_row) -> list:
    article_id = list_row[0]
    article = open(f"../data/sep11release/{LANG_PIPELINE}/raw-documents/{article_id}", "r")
    content = article.read()
    list_row[0] = re.sub(r'"|“|”', '\'', content) # Aspas especiais: “ ”

    return list_row

def add_article_title_column(list_row) -> list:
    split_article_text = list_row[0].split("\n")
    article_title = split_article_text[0]
    article_content = "\n".join(split_article_text[2:])

    list_row.insert(0, article_title)
    list_row[1] = article_content

    return list_row

def get_narratives_labels(list_row) -> list:
    domain_subnarr_list = list_row[3].split(";")

    # Add narrative_subnarrative
    narr_subnarr_list = list(map(lambda subn: ': '.join(subn.split(": ")[1:]), domain_subnarr_list))
    list_row[3] = ';'.join(narr_subnarr_list)

    # Add narrative
    domain_narr_list = list_row[2].split(";")
    narr_list = list(map(lambda narr: narr.split(": ")[1], domain_narr_list))
    list_row[2] = ';'.join(narr_list)

    # Add subnarrative
    subnarr_list = list(map(lambda narr: narr.split(": ")[2], domain_subnarr_list))
    list_row.append(';'.join(subnarr_list))

    # Change 'narr_sub' and 'subnarr' positions
    aux = list_row[3]
    list_row[3] = list_row[4]
    list_row[4] = aux

    # Add domain
    list_row.append(domain_subnarr_list[0].split(": ")[0])

    return list_row

def transform_to_csv_rows(list_row) -> str:
    transformed_row: str = ",".join(list(map(lambda e: f'"{e}"', list_row)))
    return transformed_row


if __name__ == "__main__":
    run()