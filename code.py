# -*- coding: utf-8 -*-

import os
import re
import requests
import urllib.request
import time
import io
import logging
import hashlib
import concurrent.futures
import random
import json
import unicodedata
import string
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from fake_useragent import UserAgent

import pandas as pd
import numpy as np
from tqdm import tqdm

import tiktoken
import openai

# Configure logging
logging.basicConfig(filename='crawler_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "www.ouellet.com"
full_url = "https://www.ouellet.com/fr-ca/"

# User-Agent configuration
ua = UserAgent()

class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []
    
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

def get_hyperlinks(url):
    try:
        headers = {'User-Agent': ua.random}
        with requests.get(url, timeout=30, headers=headers) as response:
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

def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link
            else:
                continue  # Ignore external links
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = f"https://{local_domain}/{link}"
        
        if clean_link:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
                
            # Ignore unwanted URLs
            if ("postulez-en-ligne" not in clean_link and 
                not re.search(r'\.(jpg|jpeg|png|gif|css|js)$', clean_link, re.IGNORECASE)):
                clean_links.append(clean_link)
    
    return list(set(clean_links))

def sanitize_filename(filename, max_length=200):
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)
    filename = filename.rstrip('.')
    
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        hash_str = hashlib.md5(filename.encode()).hexdigest()[:8]
        filename = f"{name[:max_length - len(ext) - 9]}_{hash_str}{ext}"
    
    return filename

def extract_text_from_pdf(pdf_content):
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting content from PDF: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = ""
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        text += element.get_text() + "\n"
    
    return clean_text(text)

def extract_text_alternative(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ', strip=True)

def normalize_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def process_url(url, local_domain):
    try:
        headers = {'User-Agent': ua.random}
        response = requests.get(url, timeout=30, allow_redirects=True, headers=headers)
        final_url = response.url
        
        if response.status_code == 404:
            logging.warning(f"Page not found: {url}")
            return None
        
        response.raise_for_status()
        
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

def crawl(url):
    local_domain = urlparse(url).netloc
    queue = deque([url])
    seen = set()
    
    if not os.path.exists("text/"):
        os.mkdir("text/")
    
    if not os.path.exists(f"text/{local_domain}/"):
        os.mkdir(f"text/{local_domain}/")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        while queue:
            urls_to_process = []
            for _ in range(min(5, len(queue))):
                if queue:
                    urls_to_process.append(queue.pop())
            
            future_to_url = {executor.submit(process_url, url, local_domain): url for url in urls_to_process}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                normalized_url = normalize_url(url)
                if normalized_url in seen:
                    continue
                seen.add(normalized_url)
                
                result = future.result()
                if result:
                    content_type, final_url, content = result
                    print(f"Processed: {final_url}")  # Display the URL
                    
                    filename = sanitize_filename(unquote(final_url[8:]).replace("/", "_"))
                    filepath = f'text/{local_domain}/{filename}.txt'
                    with open(filepath, "w", encoding='utf-8') as f:
                        f.write(content)
                    logging.info(f"{content_type.upper()} extracted: {final_url}")
                    
                    if content_type == 'html':
                        for link in get_domain_hyperlinks(local_domain, final_url):
                            if normalize_url(link) not in seen:
                                queue.append(link)
                
            time.sleep(random.uniform(1, 3))  # Random pause between 1 and 3 seconds
                        
    print(f"Crawling and text extraction completed for domain {local_domain}.")

# Run the crawler
crawl(full_url)

# Embeddings part

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")

# Function to split text into chunks
def split_into_chunks(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Extract the first two lines as header
    lines = text.split('\n')
    header = '\n'.join(lines[:2]) + '\n'
    
    # Remaining text
    remaining_text = '\n'.join(lines[2:])
    
    # Encode remaining text
    tokens = tokenizer.encode(remaining_text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokenizer.encode(header) + tokens[i:i + max_tokens]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
    
    return chunks

# Function to get the embedding of text
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"  # Adjust model as needed
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Folder containing .txt files
folder_path = os.path.join("text", domain)

# List of chunk sizes to process
chunk_sizes = [400, 800, 1200]

# Dictionary to store all results
all_results = {size: [] for size in chunk_sizes}

try:
    # Iterate over all .txt files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            print(f"Processing file: {filename}")
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            print(f"Content size: {len(content)} characters")
            
            # Clean the text
            cleaned_content = clean_text(content)
            
            # Process for each chunk size
            for chunk_size in chunk_sizes:
                print(f"Processing for chunk size: {chunk_size}")
                
                # Split content into chunks
                chunks = split_into_chunks(cleaned_content, chunk_size)
                
                print(f"Number of chunks: {len(chunks)}")
                
                # Get embedding for each chunk
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i+1}/{len(chunks)}")
                    embedding = get_embedding(chunk)
                    if embedding:
                        all_results[chunk_size].append({
                            'filename': filename,
                            'chunk_id': i,
                            'text': chunk,
                            'embedding': embedding
                        })
                    else:
                        print(f"Failed to obtain embedding for chunk {i+1} of file {filename}")
    
    # Create output directories and save results
    for chunk_size in chunk_sizes:
        df = pd.DataFrame(all_results[chunk_size])
        output_file = f'embeddings_results_{chunk_size}tok.csv'
        df.to_csv(output_file, index=False)
        print(f"Results for {chunk_size} tokens saved in {output_file}")
        
        # Create folder for this chunk size if it doesn't exist
        folder_name = f"{chunk_size}tok"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Prepare data for chunks.json
        chunks_list = []
        for index, row in df.iterrows():
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
        with open(os.path.join(folder_name, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(chunks_list, f, ensure_ascii=False, indent=2)
        
        # Prepare embeddings for embeddings.npy
        embeddings_list = [row['embedding'] for _, row in df.iterrows()]
        embeddings_array = np.array(embeddings_list)
        
        # Save embeddings.npy
        np.save(os.path.join(folder_name, 'embeddings.npy'), embeddings_array)
        
        print(f"Files chunks.json and embeddings.npy created successfully for {chunk_size} tokens.")
    
    print(f"Embeddings completed.")
    for chunk_size in chunk_sizes:
        print(f"Total chunks processed for {chunk_size} tokens: {len(all_results[chunk_size])}")
    
except Exception as e:
    print(f"An error occurred: {e}")
