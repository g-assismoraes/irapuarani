from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd
import gc  # Garbage collection

## based on local machine
INPUT_FILE_PATH = "merged_dataframe_with_language.csv"
OUTPUT_FILE_PATH = "translated_dataframe.csv"
INTERMEDIATE_SAVE_PATH = "translated_dataframe_intermediate.csv"
SAVE_INTERVAL = 10  # Save progress every 10 rows
RESUME_INDEX = 0  # Resume from this index

# Load the input CSV file
df = pd.read_csv(INPUT_FILE_PATH)
texts = df["content"].tolist()
languages = df["language"].tolist()

# Set up the model and tokenizer
model_id = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipeline = transformers.pipeline(
    task="text-generation",
    trust_remote_code=True,
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def get_prompt(text):
    return f'''Translate the following text into English. Be as precise as possible in retaining the information conveyed.

{text}'''

# Load intermediate results if available
translated_texts = []
if RESUME_INDEX > 0 and pd.read_csv(INTERMEDIATE_SAVE_PATH).shape[0] > 0:
    df_intermediate = pd.read_csv(INTERMEDIATE_SAVE_PATH)
    translated_texts = df_intermediate["en_content"].tolist()
    print(f"Loaded {len(translated_texts)} already translated rows.")

# Continue the translation process from the specified resume index
for idx in range(RESUME_INDEX, len(texts)):
    text = texts[idx]
    lang = languages[idx]
    print(lang)

    if lang == "EN" :
        # If already in English, copy the original text
        translated_texts[idx] = text
    if lang == "HI":
        translated_texts[idx] = text
        
    else:
        prompt = get_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        
        outputs = pipeline(
            messages,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        
        translated_text = outputs[0]["generated_text"][1]['content']
        translated_texts[idx] = translated_text
    
    # Free memory after processing each iteration
    torch.cuda.empty_cache()
    gc.collect()

    # Save intermediate results periodically
    if (idx + 1) % SAVE_INTERVAL == 0:
        # Ensure the lengths match by padding with empty strings if needed
        #current_length = len(translated_texts)
        df["en_content"] = translated_texts #+ [""] * (len(df) - current_length)
        df.to_csv(INTERMEDIATE_SAVE_PATH, index=False)
        print(f"Saved intermediate progress at row {idx + 1}")

# Add the translated texts as a new column in the DataFrame
df["en_content"] = translated_texts #+ [""] * (len(df) - len(translated_texts))

# Save the updated DataFrame to a new CSV file
df.to_csv(OUTPUT_FILE_PATH, index=False)

print(f"Translation completed. The updated CSV is saved at {OUTPUT_FILE_PATH}.")
