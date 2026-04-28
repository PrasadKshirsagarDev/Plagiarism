import pdfplumber, re, os, pandas as pd

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[।?!])\s+', text)
            if 5 <= len(s.split()) <= 40]

rows = []

for pdf in os.listdir("data/gov_pdfs"):
    if pdf.endswith(".pdf"):
        with pdfplumber.open(f"data/gov_pdfs/{pdf}") as file:
            for page in file.pages:
                text = page.extract_text()
                if text:
                    for s in split_sentences(text):
                        rows.append({
                            "sentence": s,
                            "label": "human",
                            "source": "gov_pdf"
                        })

pd.DataFrame(rows).to_csv("output/gov_marathi.csv", index=False)
print("Government PDF CSV created.")
