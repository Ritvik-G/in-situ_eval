import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
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

def build_graph(chunks, embeddings):
    """Build a graph for chunks using cosine similarity."""
    graph = nx.Graph()

    # Create nodes for each chunk
    for i in range(len(chunks)):
        graph.add_node(i, text=chunks[i], embedding=embeddings[i])

    # Add edges based on cosine similarity between chunk embeddings
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if similarity > 0.5:  # Only connect nodes with enough similarity
                graph.add_edge(i, j, weight=similarity)

    return graph

def retrieve_from_graph(graph, query_embedding, top_k=3):
    """Retrieve the most relevant contexts from the graph, using edge information."""
    similarities = []

    # Calculate cosine similarity with query_embedding
    for node, data in graph.nodes(data=True):
        node_embedding = data['embedding']
        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        similarities.append((node, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_nodes = [node for node, _ in similarities[:top_k]]

    # Use edge information to refine the top_k nodes' context relevance
    refined_contexts = []
    for node in top_k_nodes:
        neighbors = list(graph.neighbors(node))
        edge_weights = [graph[node][neighbor]['weight'] for neighbor in neighbors]

        # Choose the neighbor with the highest edge weight (cosine similarity)
        if neighbors:
            most_relevant_neighbor = neighbors[np.argmax(edge_weights)]
            refined_contexts.append(graph.nodes[most_relevant_neighbor]['text'])
        else:
            refined_contexts.append(graph.nodes[node]['text'])

    return refined_contexts


def run_model(model_config,data):
    """Run the GraphRAG model on the JSON input and add predictions to each entry"""
    
    # Iterate over each dataset (e.g., "squad", "trivia_qa", "wiki_qa")
    for dataset_key in data:
        # Iterate over each question-context pair in the dataset
        for entry in data[dataset_key]:
            question = entry['Question']
            context = entry['Context']
            
            # Generate answer using GraphRAG
            chunks = chunk_documents([context])
            embeddings = generate_embeddings(chunks)
            graph = build_graph(chunks, embeddings)
            
            # Query embedding
            query_embedding = model.encode([question])[0]
            
            # Retrieve relevant contexts. Taking first response as the retrieved context
            retrieved_info = retrieve_from_graph(graph, query_embedding)
            #retrieved = retrieved_info[0] if retrieved_info else ""

            # Generate LLM response and name it as predicted 
            ''' [This can be called in as a prompting strategy later on] '''
            prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{retrieved_info}"
            predicted = generate_response(model_config,prompt)

            # Add prediction to the entry (placed after "Response")
            entry['Predicted'] = predicted  # Insert prediction into the JSON
            
    
    return data  # Return the modified JSON with predictions