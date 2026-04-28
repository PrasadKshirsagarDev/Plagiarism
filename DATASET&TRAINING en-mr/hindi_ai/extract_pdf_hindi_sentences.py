#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect Hindi human-written text from:
  A) A single URL (--url)
  B) Hindi Wikipedia search by year + query (--year --query)

It cleans content by:
  - Removing headings, navigation, tables, scripts/styles
  - Removing bracketed refs like [1], (…)
  - Dropping very short/very long/noisy lines
  - Keeping mostly-Devanagari sentences/paragraphs

Outputs: JSONL and CSV with columns/fields: text, label (Human), source_url
"""

import re
import os
import csv
import sys
import json
import time
import argparse
from typing import List, Dict, Optional
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

HI_WIKI_API = "https://hi.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HindiTextCollector/1.0; +https://example.com/)"
}

# -----------------------------
# Utilities
# -----------------------------

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
MULTISPACE_RE = re.compile(r"\s+")
BRACKETS_RE = re.compile(r"\[[^\]]*\]")  # [1], [citation]
PARENS_RE = re.compile(r"\([^)]{0,80}\)")  # ( … ) short ref-like
CURLY_RE = re.compile(r"\{[^}]{0,120}\}")  # templates that might leak in
URL_RE = re.compile(r"https?://\S+")
BULLET_PREFIX_RE = re.compile(r"^[•\-–—\*]\s*")
HEADING_LIKE_RE = re.compile(r"^[#=•\-\*0-9\.\s]{0,6}[A-Za-z0-9०-९]+[^।]*$")

SENT_SPLIT_RE = re.compile(r"(।|\?|!)")

def is_mostly_devanagari(text: str, min_ratio: float = 0.6) -> bool:
    if not text:
        return False
    total = max(1, len(text))
    dev_count = len(DEVANAGARI_RE.findall(text))
    return (dev_count / total) >= min_ratio

def clean_line(text: str) -> str:
    text = BRACKETS_RE.sub("", text)
    text = PARENS_RE.sub("", text)
    text = CURLY_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = BULLET_PREFIX_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def looks_like_heading(line: str) -> bool:
    if len(line) <= 15:
        return True
    if "।" not in line and ("?" not in line) and ("!" not in line):
        if len(line) <= 40:
            return True
    if HEADING_LIKE_RE.match(line) and len(line) <= 50:
        return True
    return False

def split_sentences_hi(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(text)
    out = []
    buf = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        delim = parts[i+1] if i+1 < len(parts) else ""
        sent = (chunk + (delim or "")).strip()
        if sent:
            out.append(sent)
    return out

def keep_sentence(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 30:
        return False
    if len(s) > 600:
        return False
    if not is_mostly_devanagari(s, min_ratio=0.6):
        return False
    return True

# -----------------------------
# Wikipedia helpers
# -----------------------------

def hiwiki_search(query: str, limit: int = 20) -> List[Dict]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    r = requests.get(HI_WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("query", {}).get("search", [])

def hiwiki_extract_page(pageid: int) -> Dict:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "pageids": pageid,
        "inprop": "url",
        "format": "json",
    }
    r = requests.get(HI_WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    page = list(data.get("query", {}).get("pages", {}).values())[0]
    return {
        "title": page.get("title", ""),
        "extract": page.get("extract", "") or "",
        "url": page.get("fullurl", ""),
    }

def get_pageid_from_url(url: str) -> Optional[int]:
    title = url.rsplit("/", 1)[-1]
    title = unquote(title)
    params = {
        "action": "query",
        "titles": title,
        "format": "json",
    }
    r = requests.get(HI_WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    return int(next(iter(pages.keys())))

def wiki_clean_to_sentences(extract: str) -> List[str]:
    lines = [clean_line(l) for l in extract.splitlines()]
    lines = [l for l in lines if l and not looks_like_heading(l)]
    text = " ".join(lines)
    text = MULTISPACE_RE.sub(" ", text).strip()
    sents = split_sentences_hi(text)
    sents = [clean_line(s) for s in sents]
    sents = [s for s in sents if keep_sentence(s)]
    return sents

# -----------------------------
# Generic URL (news/article) extraction
# -----------------------------

def fetch_and_parse(url: str) -> str:
    hdrs = {
        "User-Agent": HEADERS["User-Agent"]
    }
    r = requests.get(url, headers=hdrs, timeout=45)
    r.raise_for_status()
    return r.text

def generic_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
        bad.decompose()
    candidates = []
    for sel in ["article", "main", "[role=main]", ".content", ".post-content", ".article__content"]:
        for node in soup.select(sel):
            candidates.append(node.get_text(" ", strip=True))
    if not candidates:
        candidates = [soup.get_text(" ", strip=True)]
    text = max(candidates, key=len) if candidates else ""
    sents = split_sentences_hi(text) or re.split(r"[\.!?]", text)
    sents = [clean_line(s) for s in sents if s]
    sents = [s for s in sents if keep_sentence(s)]
    return "\n".join(sents)

# -----------------------------
# Save
# -----------------------------

def save_jsonl_csv(rows: List[Dict], out_prefix: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    jpath = out_prefix + ".jsonl"
    cpath = out_prefix + ".csv"

    with open(jpath, "w", encoding="utf-8") as jf:
        for r in rows:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(cpath, "w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["text", "label", "source_url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[✓] Saved {len(rows)} samples")
    print(f"    JSONL: {jpath}")
    print(f"    CSV  : {cpath}")

# -----------------------------
# Main
# -----------------------------

def group_to_paragraphs(sentences: List[str], max_len: int = 400) -> List[str]:
    paras = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_len:
            buf = buf + " " + s
        else:
            paras.append(buf)
            buf = s
    if buf:
        paras.append(buf)
    return paras

def main():
    ap = argparse.ArgumentParser(
        description="Collect clean Hindi text (Human) from Wikipedia or a given URL."
    )
    ap.add_argument("--url", type=str, help="Single article URL (Wikipedia or news)")
    ap.add_argument("--year", type=str, help="Year filter for Wikipedia search (e.g., 2018)")
    ap.add_argument("--query", type=str, help="Hindi Wikipedia search query (e.g., 'भारत राजनीति')")
    ap.add_argument("--limit", type=int, default=20, help="Max Wikipedia pages to fetch")
    ap.add_argument("--out", type=str, default="human_hindi", help="Output file prefix (no extension)")
    ap.add_argument("--granularity", choices=["sentences", "paragraphs"], default="sentences",
                    help="Save as sentence-level or paragraph-level")
    args = ap.parse_args()

    rows: List[Dict] = []

    if args.url:
        args.url = unquote(args.url)  # decode Hindi
        print(f"[i] Fetching URL: {args.url}")
        html = fetch_and_parse(args.url)
        text = ""

        if "hi.wikipedia.org" in args.url:
            try:
                pageid = get_pageid_from_url(args.url)
                meta = hiwiki_extract_page(pageid)
                sents = wiki_clean_to_sentences(meta["extract"])
                if args.granularity == "paragraphs":
                    text = "\n".join(group_to_paragraphs(sents))
                else:
                    text = "\n".join(sents)
            except Exception as e:
                print(f"[!] Wikipedia API failed, fallback to generic parse: {e}")
                text = generic_article_text(html)
        else:
            text = generic_article_text(html)

        items = [t for t in text.split("\n") if keep_sentence(t)]
        for it in items:
            rows.append({"text": it, "label": "Human", "source_url": args.url})

    elif args.query:
        q = args.query
        if args.year and args.year.isdigit():
            q = f"{q} {args.year}"
        print(f"[i] Wikipedia search: '{q}' (limit={args.limit})")
        results = hiwiki_search(q, limit=args.limit)

        for i, hit in enumerate(results, 1):
            pageid = hit.get("pageid")
            title = hit.get("title", "")
            meta = hiwiki_extract_page(pageid)
            sents = wiki_clean_to_sentences(meta["extract"])
            for s in sents:
                rows.append({"text": s, "label": "Human", "source_url": meta["url"]})
            print(f"  [{i}/{len(results)}] {title} -> +{len(sents)} sentences")
            time.sleep(0.2)

    else:
        print("ERROR: Provide either --url OR --query (optionally with --year).")
        sys.exit(1)

    uniq = {}
    for r in rows:
        key = r["text"]
        if key not in uniq:
            uniq[key] = r
    rows = list(uniq.values())

    save_jsonl_csv(rows, args.out)

if __name__ == "__main__":
    main()
