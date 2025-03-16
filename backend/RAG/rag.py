from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from RAG.model_config import generate_response


# Initialize the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def chunk_documents(documents, chunk_size=100):
    """Split documents into smaller chunks."""
    chunks = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i+chunk_size])
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings for each chunk."""
    return model.encode(chunks)

def cluster_chunks(embeddings, num_clusters=5):
    """Cluster chunks using KMeans."""
    num_samples = len(embeddings)
    # Set the number of clusters to the minimum of the sample size or the desired clusters
    num_clusters = min(num_clusters, num_samples)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_

def summarize_chunks(chunks, labels):
    """Summarize each cluster."""
    summaries = {}
    for label in set(labels):
        cluster_chunks = [chunks[i] for i in range(len(chunks)) if labels[i] == label]
        # For simplicity, concatenate all chunks in a cluster
        summaries[label] = ' '.join(cluster_chunks)
    return summaries

def run_model(model_config,data):
    """Evaluate RAPTOR model on the given data. """
    # Iterate over each dataset (e.g., "squad", "trivia_qa", "wiki_qa")
    for dataset_key in data:

        # Iterate over each question-context pair in the dataset
        for entry in data[dataset_key]:
            question = entry['Question']
            context = entry['Context']

            # Generate LLM response and name it as predicted 
            ''' [This can be called in as a prompting strategy later on] '''
            prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{context}"
            predicted = generate_response(model_config,prompt)

            # Add prediction to the entry (placed after "Response")
            entry['Predicted'] = predicted  # Insert prediction into the JSON

    return data  # Return the modified JSON with predictions