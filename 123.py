import os
import shutil

# --- CONFIGURATION ---
source_folder = "swissprot_pdb_v6"
destination_folder = "swissprot_competition"

# List of names to move (e.g., extracted IDs)
with open("protein_list.txt", "r") as f:
    lines = f.readlines()
names_to_move = [line.strip() for line in lines]

# Make sure destination exists
os.makedirs(destination_folder, exist_ok=True)

# --- PROCESS FILES ---
for filename in os.listdir(source_folder):
    if filename.endswith(".pdb"):
        parts = filename.split("-")
        if len(parts) >= 3:  # ensure the format matchesp
            extracted_name = parts[1]
            if extracted_name in names_to_move:
                src = os.path.join(source_folder, filename)
                dst = os.path.join(destination_folder, filename)
                shutil.move(src, dst)