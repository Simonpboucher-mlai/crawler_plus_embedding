# -*- coding: utf-8 -*-

import os
import re
import requests
import time
import io
import logging
import hashlib
import concurrent.futures
import random
import json
import unicodedata
import string
import threading
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote, urljoin

from bs4 import BeautifulSoup
from fake_useragent import UserAgent

import pandas as pd
import numpy as np
from tqdm import tqdm

import tiktoken
import openai

# Import pdfplumber for PDF text extraction
import pdfplumber

# ===========================
# Configuration and Settings
# ===========================

# Configure logging
logging.basicConfig(
    filename='crawler_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "www.ouellet.com"
full_url = "https://www.ouellet.com/fr-ca/"

# User-Agent configuration
ua = UserAgent()

# Maximum retries for requests
MAX_RETRIES = 3

# Delay between retries in seconds
RETRY_DELAY = 5

# Maximum number of worker threads
MAX_WORKERS = 10

# Random sleep time between requests to prevent overloading server
SLEEP_TIME = (1, 3)  # Min and max seconds

# Chunk sizes configuration (nombre de tokens par chunk)
CHUNK_SIZES = [1000]  # Vous pouvez ajouter d'autres tailles si nécessaire

# ======================================
# Crawler and Text Extraction Functions
# ======================================

class HyperlinkParser(HTMLParser):
    """HTML parser to extract hyperlinks."""
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


def get_hyperlinks(url, session):
    """Fetches hyperlinks from a given URL using the provided session."""
    try:
        headers = {'User-Agent': ua.random}
        with session.get(url, timeout=30, headers=headers, allow_redirects=True) as response:
            response.raise_for_status()
            if not response.headers.get('Content-Type', '').startswith("text/html"):
                return []
            html = response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

    parser = HyperlinkParser()
    parser.feed(html)
    return parser.hyperlinks


def get_domain_hyperlinks(local_domain, url, session):
    """Filters and cleans hyperlinks to keep only those within the same domain."""
    clean_links = []
    for link in set(get_hyperlinks(url, session)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link
            else:
                continue  # Ignore external links
        else:
            if link.startswith("/"):
                clean_link = urljoin(f"https://{local_domain}", link)
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            else:
                clean_link = urljoin(url, link)

        if clean_link:
            clean_link = clean_link.rstrip('/')

            # Include PDFs and other documents
            if ("postulez-en-ligne" not in clean_link and
                not re.search(r'\.(jpg|jpeg|png|gif|css|js)$', clean_link, re.IGNORECASE)):
                clean_links.append(clean_link)

    return list(set(clean_links))


def sanitize_filename(filename, max_length=200):
    """Sanitize and truncate filenames to prevent filesystem errors."""
    filename = unquote(filename)
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    sanitized = sanitized.rstrip('.')

    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        hash_str = hashlib.md5(sanitized.encode()).hexdigest()[:8]
        sanitized = f"{name[:max_length - len(ext) - 9]}_{hash_str}{ext}"

    return sanitized or 'unnamed'


def extract_text_from_pdf(pdf_content):
    """Extracts text from PDF content, including tables."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    # Convert table to string
                    table_text = '\n'.join(['\t'.join(map(str, row)) for row in table])
                    text += table_text + "\n"
    except Exception as e:
        logging.error(f"Error extracting content from PDF: {e}")
    return text


def clean_text(text):
    """Cleans up text by removing excessive whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def extract_text_from_html(html_content):
    """Extracts meaningful text from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()

    text = "\n".join(element.get_text() for element in soup.find_all(
        ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']))
    return clean_text(text)


def extract_text_alternative(html_content):
    """Alternative method to extract text from HTML if the main method fails."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ', strip=True)


def normalize_url(url):
    """Normalizes URLs to a standard format."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def process_url(url, local_domain, session):
    """Processes a single URL: fetches content and extracts text."""
    try:
        headers = {'User-Agent': ua.random}
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = session.get(url, timeout=30, headers=headers, allow_redirects=True)
                if response.status_code == 404:
                    logging.warning(f"Page not found: {url}")
                    return None
                response.raise_for_status()
                break
            except requests.RequestException as e:
                retries += 1
                logging.warning(f"Retry {retries}/{MAX_RETRIES} for {url} due to {e}")
                time.sleep(RETRY_DELAY)
        else:
            logging.error(f"Max retries exceeded for {url}")
            return None

        final_url = response.url
        content_type = response.headers.get('Content-Type', '').lower()

        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            pdf_text = extract_text_from_pdf(response.content)
            if pdf_text.strip():
                return ('pdf', final_url, pdf_text)
            else:
                logging.warning(f"Empty PDF content: {final_url}")
        elif 'text/html' in content_type:
            text = extract_text_from_html(response.content)
            if not text.strip():
                text = extract_text_alternative(response.content)
            if text.strip():
                return ('html', final_url, text)
            else:
                logging.warning(f"Empty HTML content: {final_url}")
        else:
            logging.warning(f"Unsupported content type: {content_type} at {final_url}")

        return None

    except requests.RequestException as e:
        logging.error(f"Error processing {url}: {e}")
        return None


def crawl(start_url):
    """Main function to manage crawling of the domain starting from start_url."""
    local_domain = urlparse(start_url).netloc
    queue = deque([start_url])
    seen = set()

    output_dir = os.path.join("text", local_domain)
    os.makedirs(output_dir, exist_ok=True)

    session = requests.Session()

    def worker():
        while True:
            url = None
            with queue_lock:
                if queue:
                    url = queue.popleft()
                else:
                    break

            if url is None:
                continue

            normalized_url = normalize_url(url)
            if normalized_url in seen:
                continue
            seen.add(normalized_url)

            result = process_url(url, local_domain, session)
            if result:
                content_type, final_url, content = result
                print(f"Processed: {final_url}")

                filename = sanitize_filename(final_url[8:].replace("/", "_"))
                filepath = os.path.join(output_dir, f"{filename}.txt")
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"{content_type.upper()} extracted: {final_url}")

                if content_type == 'html':
                    links = get_domain_hyperlinks(local_domain, final_url, session)
                    with queue_lock:
                        for link in links:
                            norm_link = normalize_url(link)
                            if norm_link not in seen:
                                queue.append(link)

            time.sleep(random.uniform(*SLEEP_TIME))  # Pause between requests

    queue_lock = threading.Lock()
    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    session.close()
    print(f"Crawling and text extraction completed for domain {local_domain}.")


# =======================
# Embedding Functions
# =======================

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")

# Maximum tokens per chunk
MAX_TOKENS = 8191  # For 'text-embedding-ada-002'

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def split_into_chunks(text, max_tokens):
    """Splits text into chunks within the max_tokens limit."""
    # Extract header if any
    lines = text.split('\n')
    header_lines = []
    for line in lines:
        if line.strip() == '':
            break
        header_lines.append(line)
    header = '\n'.join(header_lines) + '\n'

    # Remaining text
    remaining_text = '\n'.join(lines[len(header_lines):])

    # Encode remaining text
    tokens = tokenizer.encode(remaining_text)
    header_tokens = tokenizer.encode(header)
    header_token_count = len(header_tokens)

    chunks = []
    max_content_tokens = max_tokens - header_token_count
    for i in range(0, len(tokens), max_content_tokens):
        chunk_tokens = header_tokens + tokens[i:i + max_content_tokens]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)

    return chunks


def get_embedding(text):
    """Gets the embedding vector for the given text."""
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None


def process_files_for_embedding(folder_path, chunk_size):
    """Processes text files to generate embeddings."""
    all_results = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for filename in tqdm(files, desc=f"Processing files with chunk size {chunk_size}"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        cleaned_content = clean_text(content)
        chunks = split_into_chunks(cleaned_content, chunk_size)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                all_results.append({
                    'filename': filename,
                    'chunk_id': i,
                    'text': chunk,
                    'embedding': embedding
                })
            else:
                logging.warning(f"Failed to get embedding for chunk {i} in file {filename}")

    return all_results


def save_embeddings(all_results, chunk_size):
    """Saves embeddings and chunks to files."""
    df = pd.DataFrame(all_results)
    output_dir = f"{chunk_size}tok"
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings dataframe
    csv_file = os.path.join(output_dir, 'embeddings.csv')
    df.to_csv(csv_file, index=False)
    print(f"Embeddings saved to {csv_file}")

    # Prepare data for chunks.json
    chunks_list = []
    for _, row in df.iterrows():
        chunk_data = {
            "text": row['text'],
            "embedding": row['embedding'],  # Stored as list
            "metadata": {
                "filename": row['filename'],
                "chunk_id": row['chunk_id']
            }
        }
        chunks_list.append(chunk_data)

    # Write chunks.json
    json_file = os.path.join(output_dir, 'chunks.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_list, f, ensure_ascii=False, indent=2)
    print(f"Chunks data saved to {json_file}")

    # Save embeddings.npy
    embeddings_array = np.array(df['embedding'].tolist())
    npy_file = os.path.join(output_dir, 'embeddings.npy')
    np.save(npy_file, embeddings_array)
    print(f"Embeddings array saved to {npy_file}")


# =======================
# Main Execution
# =======================

if __name__ == '__main__':
    # Start crawling
    crawl(full_url)

    # Folder contenant les fichiers .txt
    folder_path = os.path.join("text", domain)

    # Utiliser les tailles de chunk définies dans la configuration
    for chunk_size in CHUNK_SIZES:
        all_results = process_files_for_embedding(folder_path, chunk_size)
        save_embeddings(all_results, chunk_size)

    print("All embeddings processed and saved successfully.")
