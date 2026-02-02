import numpy as np

class ClassicalRetriever:
    """
    Moteur de recherche classique (Baseline).
    Utilise la similarité cosinus standard pour retrouver les documents.
    Sert de point de comparaison pour valider (ou invalider) l'approche quantique Grover.
    """
    def __init__(self, vectors, chunks):
        self.vectors = vectors # Base vectorielle en mémoire (numpy array)
        self.chunks = chunks # Segments de texte correspondants

    def search(self, query_text, pca_model, embedding_model, top_k=8):
        # 1. Prétraitement de la requête (Pipeline identique au quantique)
        # On applique STRICTEMENT les mêmes transformations (Embedding -> PCA) 
        # pour garantir que la comparaison "Quantique vs Classique" est équitable.
        q_emb = embedding_model.encode([query_text])
        q_red = pca_model.transform(q_emb)
        
        # Normalisation L2
        # Indispensable pour que le produit scalaire soit équivalent à la similarité Cosinus.
        q_norm = q_red / (np.linalg.norm(q_red) + 1e-10)
        
        # 2. Calcul de similarité (Algorithme "Brute Force" O(N))
        # On calcule le produit scalaire entre le vecteur requête et TOUS les documents de la base.
        # C'est rapide sur 2000 docs, mais scalerait mal sur des millions sans index (FAISS/Pinecone).
        scores = np.dot(self.vectors, q_norm.T).flatten()
        
        # 3. Récupération des meilleurs résultats (Top-K)
        # Tri décroissant des scores pour obtenir les indices les plus pertinents.
        best_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in best_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(scores[idx]), # Score de similarité (proche de 1 = très pertinent)
                "vector": self.vectors[idx], # On garde le vecteur pour l'étape de diversification (MMR)
                "index": int(idx)
            })
        return results