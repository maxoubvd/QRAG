import pennylane as qml
from pennylane import numpy as np
import pickle
import time  # <--- Ajouté car manquant dans ton import
import config

class QuantumGroverRetrieverNoise:
    """
    Version dédiée à la simulation bruitée (NISQ).
    Utilise 'default.mixed' et injecte manuellement des erreurs de dépolarisation.
    """
    def __init__(self, data_path="data/processed_data.pkl", noise_level=0.05):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.vectors = np.array(data["vectors"])
        self.chunks = data["chunks"]
        self.n_docs = len(self.vectors)
        self.n_qubits = int(np.ceil(np.log2(self.n_docs)))
        
        self.noise_level = noise_level 

        # 'default.mixed' est requis pour les canaux de bruit (matrices densité)
        # Attention : C'est beaucoup plus lent que lightning.qubit !
        self.dev = qml.device("default.mixed", wires=self.n_qubits)
        print(f"⚠️ Initialisation Simulation BRUITÉE (Niveau {self.noise_level*100}%) : {self.n_qubits} qubits.")

    def get_oracle(self, query_vector, num_to_mark):
        similarities = np.dot(self.vectors, query_vector)
        indices_to_mark = np.argpartition(similarities, -num_to_mark)[-num_to_mark:]
        for idx in indices_to_mark:
            qml.FlipSign(int(idx), wires=range(self.n_qubits))

    def get_diffusion(self):
        for w in range(self.n_qubits): qml.Hadamard(wires=w)
        qml.FlipSign(0, wires=range(self.n_qubits))
        for w in range(self.n_qubits): qml.Hadamard(wires=w)

    def apply_noise(self):
        """Applique une couche de bruit sur tous les qubits."""
        for w in range(self.n_qubits):
            qml.DepolarizingChannel(self.noise_level, wires=w)

    def search(self, query_text, pca_model, embedding_model, top_k=3):
        q_emb = embedding_model.encode([query_text])
        q_red = pca_model.transform(q_emb)
        q_norm = q_red / (np.linalg.norm(q_red) + 1e-10)
        query_vector = q_norm.flatten()
        
        n_iterations = int(np.pi / 4 * np.sqrt(self.n_docs / top_k))
        if n_iterations > 8: n_iterations = 8
        if n_iterations < 1: n_iterations = 1
        
        print(f"   -> Exécution de {n_iterations} itérations (Mode Bruité)...")

        @qml.qnode(self.dev)
        def noisy_circuit():
            # 1. Initialisation
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Bruit après l'initialisation
            self.apply_noise()

            # 2. Boucle de Grover
            for _ in range(n_iterations):
                self.get_oracle(query_vector, num_to_mark=top_k)
                # Bruit après l'Oracle (simule la décohérence pendant le calcul long)
                self.apply_noise()
                
                self.get_diffusion()
                # Bruit après la Diffusion
                self.apply_noise()
            
            return qml.probs(wires=range(self.n_qubits))
        
        # Mesure du temps (Optionnel mais utile pour comparer avec l'idéal)
        start = time.time()
        probs = noisy_circuit()
        end = time.time()

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
        
        # On retourne le vrai temps d'exécution au lieu de 0
        return results, (end - start)