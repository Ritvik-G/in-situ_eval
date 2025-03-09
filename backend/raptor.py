from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model_config import generate_response


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

def build_tree(summaries):
    """Build a hierarchical tree from summaries."""
    tree = {i: {'summary': summary} for i, summary in summaries.items()}
    return tree

def retrieve(tree, query, top_k=3):
    """Retrieve relevant information from the tree."""
    query_embedding = model.encode([query])[0]
    similarities = []
    for node, data in tree.items():
        node_embedding = model.encode([data['summary']])[0]
        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        similarities.append((node, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [tree[node]['summary'] for node, _ in similarities[:top_k]]

# Preprocess the data
def preprocess_data(dataset):
    """Preprocess the dataset into a list of tuples (question, context, response)."""
    data = []
    for item in dataset:
        question = item['Question']
        context = item['Context']
        response = item['Response']
        data.append((question, context, response))
    return data

def run_model(model_config,data):
    """Evaluate RAPTOR model on the given data. """
    # Iterate over each dataset (e.g., "squad", "trivia_qa", "wiki_qa")
    for dataset_key in data:

        # Iterate over each question-context pair in the dataset
        for entry in data[dataset_key]:
            question = entry['Question']
            context = entry['Context']

            # Generate answer using RAPTOR model
            chunks = chunk_documents([context])
            embeddings = generate_embeddings(chunks)
            labels = cluster_chunks(embeddings)
            summaries = summarize_chunks(chunks, labels)
            tree = build_tree(summaries)
            

            # Take the first retrieved answer as the tree retrieved response
            retrieved_info = retrieve(tree, question)
            
            #retrieved = retrieved_info[0] if retrieved_info else ""

            # Generate LLM response and name it as predicted 
            ''' [This can be called in as a prompting strategy later on] '''
            prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{retrieved_info}"
            predicted = generate_response(model_config,prompt)

            # Add prediction to the entry (placed after "Response")
            entry['Predicted'] = predicted  # Insert prediction into the JSON


    return data  # Return the modified JSON with predictions


