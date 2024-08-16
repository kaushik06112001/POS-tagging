import pandas as pd

# File paths
assamese_tag_path = 'assamese_tagg.txt'
pos_csv_path = 'POS_input.csv'

# Read the data from assamese_tag.txt
with open(assamese_tag_path, 'r', encoding='utf-8') as file:
    assamese_tag_data = []
    for line in file:
        # Split by tab and strip whitespace
        parts = line.strip().split('\t')
        if len(parts) == 2:
            word, pos = parts
            word = word.strip()
            pos = pos.strip().lower()  # Make the POS tags lowercase to match POS.csv format
            assamese_tag_data.append((word, pos))

# Convert the new data to a DataFrame
df_new = pd.DataFrame(assamese_tag_data, columns=['word', 'pos'])

# Read the existing POS.csv
df_existing = pd.read_csv(pos_csv_path, header=None, names=['word', 'pos'])

# Combine the existing and new data
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Save the combined data back to POS.csv
df_combined.to_csv(pos_csv_path, index=False, header=False, encoding='utf-8')

print(f"Data from {assamese_tag_path} has been successfully added to {pos_csv_path}.")
