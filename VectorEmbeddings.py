import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import torch

class VectorSearchEngine:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()  # Use eval mode
        self.index = None

    def encode(self, texts):
        """Generates embeddings for a list of texts."""
        with torch.no_grad():  # No need to track gradients
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling

    def create_index(self, embeddings):
        """Creates a FAISS index."""
        dimension = embeddings.shape[1]  # Get the dimension of embeddings
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
        self.index.add(embeddings)  # Add embeddings to the index

    def search(self, query_text, k=5):
        """Searches the index for similar texts to the query_text."""
        query_embedding = self.encode([query_text])
        distances, indices = self.index.search(query_embedding, k)  # Search for k nearest neighbors
        return distances, indices

# Example usage
if __name__ == "__main__":
    texts = ["Hello, world!", "How are you?", "The weather is great today.", "What's your favorite book?"]
    engine = VectorSearchEngine()

    embeddings = engine.encode(texts)
    engine.create_index(embeddings)

    # Let's search for texts similar to "Good day!"
    distances, indices = engine.search("Good day!")
    for idx, distance in zip(indices[0], distances[0]):
        print(f"Found: {texts[idx]}, Distance: {distance}")