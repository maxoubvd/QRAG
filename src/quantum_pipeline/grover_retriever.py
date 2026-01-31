import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import pickle
import time

# [TODO] : Adapter cette classe pour découper le dataset en lots (batches) afin de permettre l'exécution 
# sur des machines IBM Quantum réelles (souvent limitées à 127 qubits bruyants).

# [TODO] : Ajouter `qml.noise` pour simuler les erreurs de décohérence et tester la robustesse 
# de l'algorithme face à un environnement NISQ réel.

# ! A décier si on créer une nouvelle classe QuantumGroverRetrieverV2 ou si on modifie directement l'actuelle.
# En créer une nouvelle permet de garder l'ancienne pour être sûr de garder une classe fonctionnelle.

class QuantumGroverRetriever:
    def __init__(self, data_path="data/processed_data.pkl"):
        # Chargement de la base vectorielle pré-calculée
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.vectors = np.array(data["vectors"])
        self.chunks = data["chunks"]
        self.n_docs = len(self.vectors)
        self.n_qubits = int(np.ceil(np.log2(self.n_docs))) # Calcul du nombre de qubits requis : 2^n >= N_docs
        
        # Initialisation du backend 'lightning.qubit' (librairie optimisée C++)
        # Beaucoup plus rapide que 'default.qubit' pour les simulations matricielles denses.
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
            
        print(f"Initialisation : {self.n_qubits} qubits pour {self.n_docs} documents.")

    def get_oracle(self, query_vector, num_to_mark):
        """
        Oracle Quantique : Identifie les solutions valides.
        Utilise qml.FlipSign pour inverser la phase (signe) des états correspondant
        aux documents les plus similaires (calculés classiquement pour la simulation).
        """
        # 1. Identification des indices cibles (Top-K classique)
        similarities = np.dot(self.vectors, query_vector)
        indices_to_mark = np.argpartition(similarities, -num_to_mark)[-num_to_mark:]
        
        # 2. Marquage quantique (Inversion de phase : state -> -state)
        for idx in indices_to_mark:
            qml.FlipSign(int(idx), wires=range(self.n_qubits))

    def get_diffusion(self):
        """
        Opérateur de Diffusion de Grover (L'amplificateur).
        Implémentation "Atomique" : H -> Réflexion autour de 0 -> H.
        Utilise qml.FlipSign(0) qui est mathématiquement équivalent à (2|0><0| - I)
        et beaucoup plus stable numériquement que de construire les portes logiques manuellement.
        """
        # 1. Passage en base de Hadamard
        for w in range(self.n_qubits): 
            qml.Hadamard(wires=w)
            
        # 2. Réflexion autour de l'état moyen (via l'état |0...0>)
        qml.FlipSign(0, wires=range(self.n_qubits))
        
        # 3. Retour en base computationnelle (Hadamard)
        for w in range(self.n_qubits): 
            qml.Hadamard(wires=w)

    def search(self, query_text, pca_model, embedding_model, top_k=3):
        # 1. Prétraitement de la requête (Embedding -> PCA -> Normalisation)
        # La normalisation L2 est critique pour l'amplitude encoding simulé.
        q_emb = embedding_model.encode([query_text])
        q_red = pca_model.transform(q_emb)
        q_norm = q_red / (np.linalg.norm(q_red) + 1e-10)
        query_vector = q_norm.flatten()
        
        # Calcul du nombre d'itérations optimal (Approx: pi/4 * sqrt(N/k))
        n_iterations = int(np.pi / 4 * np.sqrt(self.n_docs / top_k))
        
        # OPTIMISATION PRAGMATIQUE :
        # On plafonne à 5 itérations pour garder une latence faible (< 1s) lors de la démo.
        # Cela suffit pour amplifier le signal sans risquer l'over-rotation.
        if n_iterations > 8: 
            n_iterations = 8
        # Sécurité : au moins 1 itération
        if n_iterations < 1: n_iterations = 1
        
        print(f"Optimisation : Exécution de {n_iterations} itérations de Grover...")

        # 2. Définition et Exécution du Circuit
        @qml.qnode(self.dev)
        def circuit():
            # Initialisation (Superposition uniforme de tous les documents)
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Boucle d'amplification de Grover
            for _ in range(n_iterations):
                self.get_oracle(query_vector, num_to_mark=top_k)
                self.get_diffusion()
            
            # Retourne les probabilités brutes
            return qml.probs(wires=range(self.n_qubits))

        # 3. Exécution
        print("Exécution du circuit quantique...")
        start = time.time()
        probs = circuit()
        end = time.time()
        
        # 4. Post-traitement : Récupération des meilleurs indices
        best_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for idx in best_indices:
            if idx < self.n_docs:
                results.append({
                    "text": self.chunks[idx],
                    "probability": float(probs[idx]),
                    "index": int(idx),
                    "vector": self.vectors[idx]
                })
        
        return results, (end - start)