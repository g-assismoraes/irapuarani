import re, os
import pdfplumber
from collections import defaultdict
import pandas as pd

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
            return theme
    return None

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
    current_narrative = None

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

    return taxonomies

pdf_path = "data/NARRATIVE-TAXONOMIES.pdf"  
taxonomies = extract_taxonomy_from_pdf(pdf_path)

theme_mapping = {
    'URW': 'Ukraine War label taxonomy',
    'CC': 'Climate Change label taxonomy',
    'Other': 'Other'
}

def count_narratives_in_txt(txt_file_path: str, taxonomies: dict, 
                            narrative_counts: defaultdict, 
                            subnarrative_counts: defaultdict, 
                            theme_counts: defaultdict) -> tuple:
    """
    Counts the number of occurrences of narratives, subnarratives, and themes in a .txt file.

    The file is read line by line, and themes, narratives, and subnarratives are counted
    based on the mapping provided by the extracted taxonomies.

    Args:
        txt_file_path (str): Path to the text file containing annotations.
        taxonomies (dict): Dictionary of taxonomies containing themes, narratives, and subnarratives.
        narrative_counts (defaultdict): Dictionary to store the count of narratives.
        subnarrative_counts (defaultdict): Dictionary to store the count of subnarratives.
        theme_counts (defaultdict): Dictionary to store the count of themes.

    Returns:
        tuple: Three updated dictionaries containing the count of narratives, subnarratives, and themes.
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
                        theme_counts[theme] += 1 
                    if theme and narrative_name.strip() in taxonomies[theme]:
                        narrative_counts[narrative_name.strip()] += 1
                    if subnarrative_name and (theme and narrative_name.strip() in taxonomies[theme]
                        and subnarrative_name.strip() in taxonomies[theme][narrative_name.strip()]):
                        subnarrative_counts[subnarrative_name.strip()] += 1
                except ValueError:
                    continue  # Ignore malformed lines

    return narrative_counts, subnarrative_counts, theme_counts

txt_files = [
    "data/sep11release/BG/subtask-2-annotations.txt",
    "data/sep11release/EN/subtask-2-annotations.txt",
    "data/sep11release/PT/subtask-2-annotations.txt",
    "data/oct16release/PT/subtask-2-annotations.txt",
    "data/oct16release/EN/subtask-2-annotations.txt",
    "data/oct16release/BG/subtask-2-annotations.txt",
    "data/oct16release/HI/subtask-2-annotations.txt",
]
#TODO: automatizar para pegar todos os caminhos dos `subtask-2-annotations.txt` auto

narrative_counts = defaultdict(int)
subnarrative_counts = defaultdict(int)
theme_counts = defaultdict(int)

# Count narratives, subnarratives, and themes in all .txt files
for txt_file in txt_files:
    narrative_counts, subnarrative_counts, theme_counts = count_narratives_in_txt(
        txt_file, taxonomies, narrative_counts, subnarrative_counts, theme_counts
    )

narrative_df = pd.DataFrame(list(narrative_counts.items()), columns=['Narrative', 'Count'])
subnarrative_df = pd.DataFrame(list(subnarrative_counts.items()), columns=['Subnarrative', 'Count'])
theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Count'])

narrative_df.to_csv("data/csv/subtask2_narrative_counts.csv", index=False)
subnarrative_df.to_csv("data/csv/subtask2_subnarrative_counts.csv", index=False)
theme_df.to_csv("data/csv/subtask2_theme_counts.csv", index=False)

print("\nData saved in 'subtask2_narrative_counts.csv', 'subtask2_subnarrative_counts.csv', and 'subtask2_theme_counts.csv'.")
