import os
import pandas as pd

directory = f"../data/oct25dev"  

# List to hold the data
data = []

# Traverse through the directory and its subdirectories
for subdir, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(subdir, file)
            # Extract the subdirectory name (language)
            language = os.path.basename(subdir)
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Append the data to the list
            data.append({
                "article_id": file,  # File name
                "content": content,  # File content
                "en_content": "",    # Empty column
                "language": language # Subdirectory name
            })

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_csv = "dev.csv"  # Name of the output CSV file
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"CSV file '{output_csv}' created successfully!")
