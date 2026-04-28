import pandas as pd

wiki = pd.read_csv("output/wiki_marathi.csv")
gov = pd.read_csv("output/gov_marathi.csv")

final = pd.concat([wiki, gov]).drop_duplicates()
final.to_csv("output/human_marathi_dataset.csv", index=False)

print("Final dataset ready.")
