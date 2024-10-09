import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, render_template
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load and process texts
def load_texts(author, base_folder='C:\\Users\\gianl\\OneDrive\\Desktop\\Veles Minds\\texts'):
    texts = []
    folder_path = os.path.join(base_folder, author)
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            texts.append(file.read())
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"Warning: Unable to read file {filename} with any of the attempted encodings.")
    
    # If no texts were loaded, use a default text
    if not texts:
        texts = [f"This is a placeholder text for {author}'s works. In actual deployment, ensure proper text data is provided."]
    
    return texts

# Create chunks from texts
def create_chunks(texts, chunk_size=1000, overlap=100):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# Create index
def create_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

# Retrieve relevant chunks
def retrieve_chunks(query, chunks, index, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, index)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Generate response
def generate_response(query, context, author):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer as if you were {author}:"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are {author}, providing answers based on the given context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Answer question
def answer_question(query, author, chunks, index):
    retrieved_chunks = retrieve_chunks(query, chunks, index)
    context = ' '.join(retrieved_chunks)
    answer = generate_response(query, context, author)
    return answer

# Dictionary to store chunks and index for each author
author_data = {}

# Load texts and create index for each author when the module is imported
authors = ['aristotle', 'kant', 'socrates']
for author in authors:
    print(f"Loading and processing texts for {author}...")
    texts = load_texts(author)
    chunks = create_chunks(texts)
    index = create_index(chunks)
    author_data[author] = {'chunks': chunks, 'index': index}
    print(f"Index created for {author}.")

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index_route():
    if request.method == 'POST':
        query = request.form['query']
        author = request.form['author'].lower()
        if author in author_data:
            chunks = author_data[author]['chunks']
            index = author_data[author]['index']
            answer = answer_question(query, author, chunks, index)
        else:
            answer = f"Author '{author}' not found in the database."
        return render_template('index.html', query=query, answer=answer, author=author)
    return render_template('index.html')

# For local development
if __name__ == '__main__':
    app.run(debug=True)
