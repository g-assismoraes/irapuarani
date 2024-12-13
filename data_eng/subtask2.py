import re, os
import pdfplumber
from collections import defaultdict
import pandas as pd
import glob
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("semeval2024_st2_metrics_count_themes.log"),
        logging.StreamHandler()
    ]
)

base_dir = "data/"

logging.info("Procurando arquivos de texto na estrutura de diretórios.")
txt_files = glob.glob(os.path.join(base_dir, "**", "subtask-2-annotations.txt"), recursive=True)

def getTheme(taxonomies: dict, data: str) -> str:
    """
    Extracts the main theme from a line of text in the PDF, updating the taxonomy dictionary.

    Args:
        taxonomies (dict): Dictionary mapping themes to their narratives and subnarratives.
        data (str): Line of text extracted from the PDF.

    Returns:
        str: Main theme of the narrative, or None if no theme is found.
    """
    for line in data.split('\n'):
        line = line.strip()
        if "Figure" in line:
            theme = line.split(':')[1].strip()
            if theme not in taxonomies:
                taxonomies[theme] = defaultdict(list)
            logging.info(f"Tema encontrado: {theme}")
            return theme
    return None

def get_language_from_filename(filename: str) -> str:
    """
    Extracts the language from the filename.
    Args:
        filename (str): The name of the file.
    Returns:
        str: The language code extracted from the filename.
    """
    match = re.search(r"(BG|EN|PT|HI|RU)", filename)
    return match.group(1) if match else "Unknown"

def extract_taxonomy_from_pdf(pdf_path: str) -> defaultdict:
    """
    Extracts the taxonomy of narratives and subnarratives from a PDF file.

    The PDF is processed page by page, identifying main themes, narratives, 
    and subnarratives, and organizing them into a hierarchical dictionary.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        defaultdict: Dictionary containing themes, narratives, and subnarratives extracted from the PDF.
                     Structure: {theme: {narrative: [subnarrative_1, subnarrative_2, ...]}}.
    """
    taxonomies = defaultdict(lambda: defaultdict(list))
    with pdfplumber.open(pdf_path) as pdf:
        theme = None
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            theme = getTheme(taxonomies, text) or theme
            for line in text.split('\n'):
                line = line.strip()
                if line == "Other":
                    current_narrative = line
                if "Figure" in line:
                    break
                if line and not line.startswith("-"):
                    current_narrative = line
                    taxonomies[theme][current_narrative].append('Other')
                elif line.startswith("-") and current_narrative:
                    subnarrative = line.lstrip("- ").strip()
                    taxonomies[theme][current_narrative].append(subnarrative)
    logging.info("Taxonomias extraídas do PDF com sucesso.")
    return taxonomies

pdf_path = "data/NARRATIVE-TAXONOMIES.pdf"  
logging.info(f"Extraindo taxonomias do arquivo PDF: {pdf_path}")
taxonomies = extract_taxonomy_from_pdf(pdf_path)

theme_mapping = {
    'URW': 'Ukraine War label taxonomy',
    'CC': 'Climate Change label taxonomy',
    'Other': 'Other'
}

def count_narratives_in_txt(txt_file_path: str, taxonomies: dict, 
                            narrative_counts: defaultdict, 
                            subnarrative_counts: defaultdict, 
                            theme_counts: defaultdict, 
                            lang: str) -> tuple:
    """
    Updated to count narratives, subnarratives, and themes, also adding clustering by language.

    Args:
        txt_file_path (str): Path to the text file containing annotations.
        taxonomies (dict): Dictionary of taxonomies containing themes, narratives, and subnarratives.
        narrative_counts (defaultdict): Dictionary to store the count of narratives by language.
        subnarrative_counts (defaultdict): Dictionary to store the count of subnarratives by language.
        theme_counts (defaultdict): Dictionary to store the count of themes by language.
        lang (str): Language of the current file.

    Returns:
        tuple: Updated dictionaries with counts per language.
    """
    with open(txt_file_path, 'r') as file:
        for line in file:
            parts = re.split('\t|;|\n', line)[:-1]
            if len(parts) < 2:
                continue

            article_id = parts[0]
            NarrativesSubnarratives_lst = parts[1:]

            for narrative in NarrativesSubnarratives_lst:
                try:
                    subnarrative_name = None
                    if len(narrative.split(':', 2)) == 3:
                        theme_abbr, narrative_name, subnarrative_name = narrative.split(':', 3)
                    elif len(narrative.split(':', 2)) == 2:
                        theme_abbr, narrative_name = narrative.split(':', 2)
                    else:
                        theme_abbr, narrative_name = (narrative, narrative)

                    theme = theme_mapping.get(theme_abbr.strip())
                    if theme:
                        theme_counts[(theme, lang)] += 1 
                    if theme and narrative_name.strip() in taxonomies[theme]:
                        narrative_counts[(narrative_name.strip(), lang)] += 1
                    if subnarrative_name and (theme and narrative_name.strip() in taxonomies[theme]
                        and subnarrative_name.strip() in taxonomies[theme][narrative_name.strip()]):
                        subnarrative_counts[(subnarrative_name.strip(), lang)] += 1
                except ValueError:
                    logging.warning(f"Linha malformada no arquivo: {txt_file_path}")
                    continue
    logging.info(f"Processado arquivo: {txt_file_path}")
    return narrative_counts, subnarrative_counts, theme_counts



narrative_counts = defaultdict(int)
subnarrative_counts = defaultdict(int)
theme_counts = defaultdict(int)

for txt_file in txt_files:
    lang = get_language_from_filename(txt_file)
    logging.info(f"Processando arquivo: {txt_file} (Idioma: {lang})")
    narrative_counts, subnarrative_counts, theme_counts = count_narratives_in_txt(
        txt_file, taxonomies, narrative_counts, subnarrative_counts, theme_counts, lang
    )

narrative_df = pd.DataFrame(
    [{'Narrative': k[0], 'Language': k[1], 'Count': v} for k, v in narrative_counts.items()]
)
subnarrative_df = pd.DataFrame(
    [{'Subnarrative': k[0], 'Language': k[1], 'Count': v} for k, v in subnarrative_counts.items()]
)
theme_df = pd.DataFrame(
    [{'Theme': k[0], 'Language': k[1], 'Count': v} for k, v in theme_counts.items()]
)

def find_zero_counts(taxonomies, counts, level):
    """
    Finds items with zero counts in the given level (theme/narrative/subnarrative).
    """
    zero_count_items = []
    for theme, narratives in taxonomies.items():
        for narrative, subnarratives in narratives.items():
            if level == 'narrative' and (narrative, "Unknown") not in counts:
                zero_count_items.append({'Theme': theme, 'Narrative': narrative, 'Count': 0})
            if level == 'subnarrative':
                for subnarrative in subnarratives:
                    if (subnarrative, "Unknown") not in counts:
                        zero_count_items.append({'Theme': theme, 'Narrative': narrative, 'Subnarrative': subnarrative, 'Count': 0})
    return zero_count_items

zero_narratives = find_zero_counts(taxonomies, narrative_counts, 'narrative')
zero_subnarratives = find_zero_counts(taxonomies, subnarrative_counts, 'subnarrative')

zero_narratives_df = pd.DataFrame(zero_narratives)
zero_subnarratives_df = pd.DataFrame(zero_subnarratives)

def generate_percentage_csv(theme_df):
    """
    Gera um CSV com as porcentagens de CC, URW e ALL por idioma.

    Args:
        theme_df (pd.DataFrame): DataFrame contendo os temas, idiomas e contagens.

    Returns:
        None
    """
    relevant_themes = ["Climate Change label taxonomy", "Ukraine War label taxonomy"]
    filtered_df = theme_df[theme_df["Theme"].isin(relevant_themes)]

    total_by_language = filtered_df.groupby("Language")["Count"].sum()

    percentages_df = filtered_df.groupby(["Language", "Theme"])["Count"].sum().unstack(fill_value=0)
    percentages_df["ALL"] = total_by_language
    percentages_df["CC (%)"] = (percentages_df["Climate Change label taxonomy"] / percentages_df["ALL"]) * 100
    percentages_df["URW (%)"] = (percentages_df["Ukraine War label taxonomy"] / percentages_df["ALL"]) * 100

    output_path = "data/csv/subtask2_theme_percentages_by_language.csv"
    percentages_df[["CC (%)", "URW (%)", "ALL"]].to_csv(output_path, index=True)
    logging.info(f"Arquivo CSV com porcentagens salvo em: {output_path}")



logging.info("Salvando resultados em arquivos CSV.")

generate_percentage_csv(theme_df)
narrative_df.to_csv("data/csv/subtask2_narrative_counts_by_language.csv", index=False)
subnarrative_df.to_csv("data/csv/subtask2_subnarrative_counts_by_language.csv", index=False)
theme_df.to_csv("data/csv/subtask2_theme_counts_by_language.csv", index=False)
zero_narratives_df.to_csv("data/csv/subtask2_zero_narratives.csv", index=False)
zero_subnarratives_df.to_csv("data/csv/subtask2_zero_subnarratives.csv", index=False)
logging.info("Processamento concluído e arquivos CSV gerados.")