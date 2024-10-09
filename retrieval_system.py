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
def load_texts(folder_path="texts"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    texts = []
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
def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Aristotle, providing answers based on the given context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Answer question
def answer_question(query):
    retrieved_chunks = retrieve_chunks(query, chunks, index)
    context = ' '.join(retrieved_chunks)
    answer = generate_response(query, context)
    return answer

# Load texts and create index when the module is imported
print("Loading and processing texts...")
texts = load_texts()
chunks = create_chunks(texts)
index = create_index(chunks)
print("Index created.")

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index_route():
    if request.method == 'POST':
        query = request.form['query']
        answer = answer_question(query)
        return render_template('index.html', query=query, answer=answer)
    return render_template('index.html')

# For local development
if __name__ == '__main__':
    app.run(debug=True)
