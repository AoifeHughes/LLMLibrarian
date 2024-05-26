import os
import openai
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import re
import sqlite3
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import Levenshtein
from tqdm import tqdm



os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Librarian:
    def __init__(self, base_url, api_key, model_name="sentence-transformers/all-mpnet-base-v2", db_name="github.db", system_prompt=None, user_prompt=None, chunk_size=256, overlap=100, num_chunks=3):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.db_name = db_name
        self.create_database()
        self.system_prompt = system_prompt or "You are a GitHub repository reviewer. Your task is to answer the following question based on the provided context from the repository. In your response, clearly state the file(s) where you found the relevant information. If the answer is present in multiple files, list all the filenames. Be concise and specific in your response."
        self.user_prompt = user_prompt or "Query: {query}\nContext:\n{context}"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_chunks = num_chunks
        self.common_file_names = ["readme.md", "code_of_conduct.md", "contributing.md", "license.md", "README.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "LICENSE.md", "readme.txt", "code_of_conduct.txt", "contributing.txt", "license.txt", "README.txt", "CODE_OF_CONDUCT.txt", "CONTRIBUTING.txt", "LICENSE.txt", "readme", "code_of_conduct", "contributing", "license", "README", "CODE_OF_CONDUCT", "CONTRIBUTING", "LICENSE"]


    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                chunks TEXT,
                embeddings TEXT,
                summary TEXT,
                summary_embedding TEXT
            )
        """)
        conn.commit()
        conn.close()


    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_text_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text = self.remove_markdown_syntax(text)
        return text
    
    def remove_markdown_syntax(self, markdown_text):
        plain_text = re.sub(r'^#.*$', '', markdown_text, flags=re.MULTILINE)
        plain_text = re.sub(r'(\*{1,2}|_{1,2})(.*?)\1', r'\2', plain_text)
        plain_text = re.sub(r'\[.*?\]\(.*?\)', '', plain_text)
        plain_text = re.sub(r'!\[.*?\]\(.*?\)', '', plain_text)
        plain_text = re.sub(r'```[^`]*```', '', plain_text, flags=re.DOTALL)
        plain_text = re.sub(r'`[^`]*`', '', plain_text)
        plain_text = re.sub(r'^\s*>\s*', '', plain_text, flags=re.MULTILINE)
        plain_text = re.sub(r'^[-*_]{3,}$', '', plain_text, flags=re.MULTILINE)
        plain_text = re.sub(r'^\|.*\|$', '', plain_text, flags=re.MULTILINE)
        plain_text = re.sub(r'\[\^.*?\]', '', plain_text)
        plain_text = re.sub(r'^\[.*?\]:.*$', '', plain_text, flags=re.MULTILINE)
        plain_text = re.sub(r'<[^>]*>', '', plain_text)
        return plain_text.strip()

    def preprocess_text(self, text):
        text = re.sub(r'(header|footer|references).*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def generate_file_summary(self, file_path, text, chunk_size=1000):
        plain_text = self.remove_markdown_syntax(text)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        chunks = [plain_text[i:i+chunk_size] for i in range(0, len(plain_text), chunk_size)]
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        combined_summary = " ".join(summaries)
        final_summary = summarizer(combined_summary[:1024], max_length=300, min_length=50, do_sample=False)

        return final_summary[0]['summary_text']

    def generate_file_summary_custom_model(self, file_path, text, chunk_size=4000):
        # Remove Markdown syntax from the text
        # Split the plain text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Generate summary for each chunk
        summaries = []
        for chunk in chunks:
            system = {
                "role": "system",
                "content": f"Please provide a concise summary of the contents of the file: {file_path}"
            }
            user = {
                "role": "user",
                "content": chunk
            }
            completion = self.client.chat.completions.create(
                model="llama",
                messages=[
                    system,
                    user
                ]
            )
            summary = completion.choices[0].message.content
            summaries.append(summary)

        # Combine the summaries of all chunks
        combined_summary = " ".join(summaries)

        # Generate a final summary of the combined summaries
        system = {
            "role": "system",
            "content": f"Please provide a concise overall summary of the contents of the file: {file_path}"
        }
        user = {
            "role": "user",
            "content": combined_summary
        }
        completion = self.client.chat.completions.create(
            model="llama",
            messages=[
                system,
                user
            ]
        )

        final_summary = completion.choices[0].message.content

        return final_summary

    def create_summary_embedding(self, summary):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        inputs = self.tokenizer(summary, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

    def calculate_file_name_similarity(self, query, file_name):
        if file_name.lower() in self.common_file_names:
            return 1.0
        query_words = query.lower().split()
        file_name_words = file_name.lower().split('_')
        
        max_similarity = 0
        for query_word in query_words:
            for file_name_word in file_name_words:
                similarity = 1 - Levenshtein.distance(query_word, file_name_word) / max(len(query_word), len(file_name_word))
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity

    def split_text_into_chunks(self, text):
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk = ' '.join(words[i:i + self.chunk_size])
                chunks.append(chunk)
            return chunks

    def find_relevant_chunks(self, query, file_name_weight=0.7):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        query_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=256)
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, chunks, embeddings FROM documents")
        results = cursor.fetchall()
        conn.close()

        all_chunks = []
        all_embeddings = []
        file_paths = []
        for file_path, chunks, embeddings in results:
            all_chunks.extend(json.loads(chunks))
            all_embeddings.extend(json.loads(embeddings))
            file_paths.extend([file_path] * len(json.loads(chunks)))

        similarities = cosine_similarity([query_embedding], all_embeddings)[0]

        file_name_similarities = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_name_similarity = self.calculate_file_name_similarity(query, file_name)
            file_name_similarity += 1 - (len(file_name) / 100)  # Prioritize shorter filenames
            file_name_similarities.append(file_name_similarity)

        combined_similarities = (1 - file_name_weight) * similarities + file_name_weight * np.array(file_name_similarities)

        sorted_indices = sorted(range(len(combined_similarities)), key=lambda i: combined_similarities[i], reverse=True)
        relevant_chunks = []
        relevant_file_paths = []
        for i in sorted_indices[:self.num_chunks]:
            relevant_chunks.append(all_chunks[i])
            relevant_file_paths.append(file_paths[i])
        return relevant_chunks, relevant_file_paths

        
    def generate_answer(self, query, relevant_chunks, relevant_file_paths):
        system = {
            "role": "system",
            "content": self.system_prompt
        }
        context = "\n".join(f"{i+1}. {chunk} (File: {file_path})" for i, (chunk, file_path) in enumerate(zip(relevant_chunks, relevant_file_paths)))
        user = {
            "role": "user",
            "content": self.user_prompt.format(query=query, context=context)
        }
        print(system)
        print(user)

        completion = self.client.chat.completions.create(
            model="llama",
            messages=[
                system,
                user
            ]
        )
        answer = completion.choices[0].message.content
        return answer
    def reset_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
        conn.close()
        self.create_database()


    def create_embeddings(self, chunks):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)

        def process_chunk(chunk):
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return embedding

        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(process_chunk, chunks))

        return embeddings

    def process_files_in_directory(self, directory):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), directory)
                file_paths.append(file_path)
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            cursor.execute("SELECT COUNT(*) FROM documents WHERE file_path = ?", (file_path,))
            if cursor.fetchone()[0] > 0:
                continue
            
            full_path = os.path.join(directory, file_path)
            if file_path.endswith(".pdf"):
                text = self.extract_text_from_pdf(full_path)
            elif file_path.endswith((".py", ".md", ".txt", ".rst", "LICENSE")):
                text = self.extract_text_from_file(full_path)
            else:
                continue
            
            preprocessed_text = self.preprocess_text(text)
            chunks = self.split_text_into_chunks(preprocessed_text)
            embeddings = self.create_embeddings(chunks)
            summary = self.generate_file_summary(file_path, preprocessed_text)
            print(f"Summary for {file_path}: {summary}")
            summary_embedding = self.create_summary_embedding(summary)
            
            cursor.execute("""
                INSERT INTO documents (file_path, chunks, embeddings, summary, summary_embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, json.dumps(chunks), json.dumps(embeddings), summary, json.dumps(summary_embedding)))
        
        conn.commit()
        conn.close()

    def query(self, question):
        print("fetching chunks...")
        relevant_chunks, relevant_file_paths = self.find_relevant_chunks(question)
        print("sending to server...")
        answer = self.generate_answer(question, relevant_chunks, relevant_file_paths)
        return answer
