import json, os, re, pandas as pd

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[।?!])\s+', text)
            if 5 <= len(s.split()) <= 40]

rows = []

for root, _, files in os.walk("data/wiki_extracted"):
    for f in files:
        if f.endswith(".json"):
            with open(os.path.join(root, f), encoding="utf-8") as file:
                for line in file:
                    article = json.loads(line)
                    for s in split_sentences(article.get("text", "")):
                        rows.append({
                            "sentence": s,
                            "label": "human",
                            "source": "wikipedia"
                        })

pd.DataFrame(rows).to_csv("output/wiki_marathi.csv", index=False)
print("Wikipedia CSV created.")
