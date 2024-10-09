import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Load and process texts
def load_texts(folder_path="C:\\Users\\gianl\\OneDrive\\Desktop\\Veles Minds\\texts"):
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
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
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

# Main execution
if __name__ == "__main__":
    folder_path = "C:\\Users\\gianl\\OneDrive\\Desktop\\Veles Minds\\texts"
    
    # Load texts and create index
    print("Loading and processing texts...")
    texts = load_texts(folder_path)
    chunks = create_chunks(texts)
    index = create_index(chunks)
    
    print("Index created.")
    
    # Save index and chunks
    print("Saving index and chunks...")
    np.save('index.npy', index)
    with open('chunks.json', 'w') as f:
        json.dump(chunks, f)
    print("Index and chunks saved.")
    
    # Interactive Q&A loop
    print("Welcome to the Aristotle Q&A system. Type 'exit' to quit.")
    while True:
        user_query = input("\nWhat question would you like to ask Aristotle? ")
        if user_query.lower() == 'exit':
            break
        answer = answer_question(user_query)
        print(f"\nAristotle: {answer}")

    print("Thank you for using the Aristotle Q&A system. Goodbye!")