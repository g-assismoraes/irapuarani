import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from datasets import Dataset
import numpy as np
import sklearn
from huggingface_hub import login

# ==========================
# Configuration Variables (based on the local machine)
# ==========================
HUGGING_FACE_TOKEN = ""
HUB_MODEL_ID_BASE = "g-assismoraes/mdeberta-domain"  # Base model ID for Hugging Face Hub
INPUT_FILE_PATH = "translated_dataframe.csv"  # Input file path
OUTPUT_DIR = "./domain_results"  # Output directory for training
NUM_FOLDS = 5  # Number of folds for cross-validation
SEED = 42  # Random seed for reproducibility
MODEL_NAME = "microsoft/deberta-v3-base"  # Pre-trained model name
NUM_EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 16  # Batch size for training and evaluation

# Login to Hugging Face Hub
login(token=HUGGING_FACE_TOKEN)

# Load the DataFrame
df = pd.read_csv(INPUT_FILE_PATH)

# Prepare the text and labels
texts = df["content"].tolist()
# Convert 'domain' into three categories: 'URW', 'CC', and 'Other'
df['label'] = df['domain'].apply(lambda x: 0 if x == "URW" else 1 if x == "CC" else 2)
labels = df['label'].tolist()

NUM_LABELS = 3  # For ternary classification: URW, CC, Other

# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to tokenize the input text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# To store metrics for all folds
all_fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
    
    print(f"Starting fold {fold}/{NUM_FOLDS}")
    
    # Split data for the current fold
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Create Dataset objects
    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    
    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/fold{fold}",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=f"{HUB_MODEL_ID_BASE}_fold{fold}",
        seed=SEED,
        save_total_limit=1,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10
    )

    # Define a function to compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        precision = sklearn.metrics.precision_score(labels, predictions, average='macro', zero_division=1)
        recall = sklearn.metrics.recall_score(labels, predictions, average='macro', zero_division=1)
        f1 = sklearn.metrics.f1_score(labels, predictions, average='macro', zero_division=1)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Push the model for the current fold to Hugging Face Hub
    trainer.push_to_hub()

    # Evaluate and get predictions
    predictions = trainer.predict(val_dataset).predictions
    predicted_labels = np.argmax(predictions, axis=-1)  # Predicted domain labels (p_domains)

    # Add predictions to the validation DataFrame for the current fold
    val_df = df.iloc[val_idx].copy()
    val_df["p_domains"] = predicted_labels

    # Save the DataFrame with predictions for the current fold
    val_df.to_csv(f"{OUTPUT_DIR}/fold{fold}_predictions.csv", index=False)

    # Store metrics
    fold_metrics = trainer.evaluate()
    all_fold_metrics.append(fold_metrics)
    print(f"Fold {fold} completed and results saved.")

# ==========================
# Save Metrics to a Log File
# ==========================
metrics_log_path = f"{OUTPUT_DIR}/cross_validation_metrics_log.txt"
with open(metrics_log_path, "w") as log_file:
    # Write individual fold metrics
    for fold, metrics in enumerate(all_fold_metrics, start=1):
        log_file.write(f"Fold {fold} Metrics:\n")
        for metric_name, value in metrics.items():
            log_file.write(f"{metric_name}: {value:.4f}\n")
        log_file.write("\n")

    # Calculate and write average metrics across all folds
    avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
    log_file.write("Average Metrics Across All Folds:\n")
    for metric_name, value in avg_metrics.items():
        log_file.write(f"{metric_name}: {value:.4f}\n")

print(f"Metrics log saved to {metrics_log_path}")
