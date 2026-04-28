import csv
import re

INPUT_FILE = "text.txt"
OUTPUT_FILE = "human_dataset3.csv"

# Read file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize whitespace
text = text.replace("\n", " ")
text = re.sub(r'\s+', ' ', text)

# Remove reference markers like [३१], [32]
text = re.sub(r'\[[०-९0-9]+\]', '', text)

# 🔥 IMPORTANT: split on all possible sentence separators
sentences = re.split(r'[।\.?!|]+', text)

clean_sentences = []

for sent in sentences:
    sent = sent.strip()

    if not sent:
        continue

    word_count = len(sent.split())

    # Relaxed filtering (VERY IMPORTANT)
    if word_count >= 3:
        clean_sentences.append(sent + "।")

print("Total raw splits:", len(sentences))
print("Total clean sentences:", len(clean_sentences))

# Write CSV
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sentence", "label"])

    for sent in clean_sentences:
        writer.writerow([sent, "human"])

print(f"✅ CSV created successfully: {OUTPUT_FILE}")
