import os
import subprocess

dump_path = "data/wiki_dump/mrwiki-latest-pages-articles.xml.bz2"
output_dir = "data/wiki_extracted"

os.makedirs(output_dir, exist_ok=True)

subprocess.run([
    "python", "-m", "wikiextractor.WikiExtractor",
    dump_path,
    "-o", output_dir,
    "--json",
    "--no-templates"
])

print("Wikipedia extraction completed.")
