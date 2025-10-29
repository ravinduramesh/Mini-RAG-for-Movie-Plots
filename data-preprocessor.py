"""
Select first 500 movies and create preprocessed CSV from raw movie plots CSV.
"""
import os
import hashlib
from typing import List

from utils import get_logger, sanitize_text

import pandas as pd
import spacy

logger = get_logger()


def generate_chunk_id(title: str, index: int) -> str:
    return hashlib.md5(f"{title}_{index}".encode("utf-8")).hexdigest()[:12]


class SpacyTextChunkerSimple:
    def __init__(self, chunk_size: int = 250, overlap: int = 50, model: str = "en_core_web_sm"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.nlp = spacy.load(model)
        except OSError:
            os.system(f"python -m spacy download {model}")  # Download the model if it is missing
            self.nlp = spacy.load(model)

    def chunk_text(self, plot: str, title: str) -> List[dict]:
        if not plot:
            return []

        doc = self.nlp(plot)
        sentences = list(doc.sents)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_word_count = 0
        chunk_index = 0

        for sent in sentences:
            sent_text = sent.text.strip()
            sent_word_count = len(sent_text.split())

            if current_word_count + sent_word_count > self.chunk_size and current_sentences:
                chunked_plot = " ".join(current_sentences)
                chunks.append({
                    "chunk_id": generate_chunk_id(title, chunk_index),
                    "title": title,
                    "chunk_index": chunk_index,
                    "chunked_plot": chunked_plot,
                    "word_count": current_word_count
                })

                # Overlap to share context between chunks
                overlap_word_count = 0
                overlap_sentences = []
                for i in range(len(current_sentences) - 1, -1, -1):
                    sent_words = len(current_sentences[i].split())
                    if overlap_word_count + sent_words <= self.overlap:
                        overlap_sentences.insert(0, current_sentences[i])
                        overlap_word_count += sent_words
                    else:
                        break

                current_sentences = overlap_sentences + [sent_text]
                current_word_count = overlap_word_count + sent_word_count
                chunk_index += 1
            else:
                current_sentences.append(sent_text)
                current_word_count += sent_word_count

        if current_sentences:
            chunked_plot = " ".join(current_sentences)
            chunks.append({
                "chunk_id": generate_chunk_id(title, chunk_index),
                "title": title,
                "chunk_index": chunk_index,
                "chunked_plot": chunked_plot,
                "word_count": current_word_count
            })

        return chunks


def main():
    df = pd.read_csv("wiki_movie_plots_deduped.csv", encoding="utf-8", on_bad_lines="skip")

    df = df.dropna(subset=["Title", "Plot"]).copy()
    df = df[df["Plot"].str.len() > 100]
    df = df.head(500)  # Limit to first 500 movies for processing

    chunker = SpacyTextChunkerSimple()

    rows = []
    for _, row in df.iterrows():
        title = sanitize_text(row["Title"])
        plot = sanitize_text(row["Plot"])
        chunks = chunker.chunk_text(plot, title)
        rows.extend(chunks)

    out_df = pd.DataFrame(rows)

    cols = ["chunk_id", "title", "chunk_index", "chunked_plot", "word_count"]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = None

    out_df = out_df[cols]
    out_df.to_csv("preprocessed_wiki_movies.csv", index=False, encoding="utf-8")
    logger.info(f"Wrote {len(out_df)} chunks to preprocessed_wiki_movies.csv")


if __name__ == "__main__":
    main()
