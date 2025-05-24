import os
import faiss
import pickle
import numpy as np

from crawler.fetch import fetch_url
from crawler.parser import parse_html
from utils.logger import get_logger
from config import (
    MAX_WORDS_PER_CHUNK,
    VECTOR_DB_PATH,
    DOC_STORE_PATH,
    EMBEDDING_MODEL_NAME,
)

from sentence_transformers import SentenceTransformer


logger = get_logger("vector_builder")


def split_text(text: str, max_words:int = MAX_WORDS_PER_CHUNK,) -> list[str]:
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


def process_url(url: str, embed_model: SentenceTransformer) -> list:
    try:
        logger.info(f"Fetching URL: {url}")
        html = fetch_url(url)
        text = parse_html(html)
        chunks = split_text(text)
        if not chunks:
            logger.warning(f"No usable content extracted from {url}")
            return []
        embeddings = embed_model.encode(chunks)
        return list(zip(chunks, embeddings))
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return []


def load_urls_from_file(file_path: str)->list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def build_vectorstore(urls: list[str]):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_chunks = []
    all_embeddings = []

    for url in urls:
        results = process_url(url, model)
        if results:
            chunks, embeds = zip(*results)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeds)
    
    if not all_chunks:
        logger.warning(f"No content processed, Vectorstore will not be saved.")
        return
    
    dim = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(all_embeddings))

    os.makedirs(VECTOR_DB_PATH.parent, exist_ok=True)
    faiss.write_index(index, str(VECTOR_DB_PATH))

    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    
    logger.info(f"Vectorstore saved to {VECTOR_DB_PATH}, {len(all_chunks)} chunks saved to {DOC_STORE_PATH}")

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Build vectorstore from one or more URLs.")

#     parser.add_argument("urls", nargs="*", help="List of URLs to process.")
#     parser.add_argument("--file", help="Path to file containing list of URLs to process.")

#     args = parser.parse_args()

#     urls = []
#     if args.file:
#         urls.extend(load_urls_from_file(args.file))
#     urls.extend(args.urls)

#     if not urls:
#         print("No URLs found in arg list.\nUsage: python scripts/build_vectorstore.py <URL1> <URL2> ... OR --file path/to/urls.txt")
#     else:
#         main(urls)
