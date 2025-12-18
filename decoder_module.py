
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ollama_module import LLMModel
import nltk

# Ensure nltk is downloaded (basic tokenizer)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class ResumeDecoder:
    def __init__(self):
        """
        A RAG-based decoder that reconstructs a resume by selecting sentences
        that best align with the target embedding.
        """
        pass

    def decode(self, target_vector, original_resume_text, top_k=10):
        """
        Selects sentences from `original_resume_text` that are semantically closest 
        to `target_vector`.

        Args:
            target_vector (numpy.ndarray or torch.Tensor): The output vector from MLP (4096,).
            original_resume_text (str): The full text of the original resume.
            top_k (int): Number of sentences to select.

        Returns:
            str: Reconstructed resume text.
        """
        # normalize target vector
        if isinstance(target_vector, torch.Tensor):
            target_vector = target_vector.detach().cpu().numpy()
        
        target_vector = target_vector.flatten().reshape(1, -1)

        # 1. Split resume into candidate sentences
        sentences = nltk.sent_tokenize(original_resume_text)
        # Clean sentences (remove short ones)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Error: No valid sentences found in resume."

        # 2. Embed all sentences
        # Note: This is slow if done sequentially. 
        # Ideally, we would batch this or cache it.
        # For a single resume (20-50 sentences), this is acceptable (few seconds).
        sent_vectors = []
        valid_sentences = []
        
        # Instantiate Client
        client = LLMModel(model_name="mxbai-embed-large")

        for sent in sentences:
            vec = client.get_vector(sent)
            if vec is not None:
                sent_vectors.append(vec)
                valid_sentences.append(sent)
        
        if not sent_vectors:
            return "Error: Could not embed sentences."

        sent_matrix = np.array(sent_vectors) # (Num_Sentences, 4096)

        # 3. Compute Similarity: Target (1, 4096) x Matrix.T (4096, N)
        similarities = cosine_similarity(target_vector, sent_matrix)[0] # Shape (N,)

        # 4. Select Top-K
        # Get indices of top-k scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Sort indices to maintain original narrative flow (optional but recommended)
        top_indices = sorted(top_indices)

        selected_sentences = [valid_sentences[i] for i in top_indices]
        
        return "\n\n".join(selected_sentences)
