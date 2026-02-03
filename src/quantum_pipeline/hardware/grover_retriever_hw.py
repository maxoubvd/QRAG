import os
# Protection des variables d'environnement
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
import pennylane as qml
from pennylane import numpy as np # Important : Numpy de PennyLane
import numpy as real_np # Numpy classique pour les manipulations de listes
import pickle
import time
import math
import config
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumGroverRetriever:
    def __init__(self, data_path="data/processed_data.pkl", batch_size=32):
        """
        Initialise le retriever pour le hardware réel.
        batch_size=32 signifie qu'on utilise 5 qubits (2^5=32) par circuit.
        C'est un bon compromis pour limiter la profondeur du circuit et le bruit.
        """
        # Chargement des données
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.vectors = real_np.array(data["vectors"])
        self.chunks = data["chunks"]
        self.total_docs = len(self.vectors)
        
        self.batch_size = batch_size
        # Nombre de qubits nécessaires pour un batch (ex: log2(32) = 5)
        self.n_qubits_batch = int(math.ceil(math.log2(batch_size)))
        
        # Initialisation du Backend IBM Quantum (Qiskit Runtime)
        # On initialise le device UNE SEULE FOIS pour la taille du batch.
        print(f"Connexion au backend IBM Quantum : {config.HARDWARE_BACKEND_NAME}...")
        
        try:
            # 1. ON SE CONNECTE MANUELLEMENT D'ABORD
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=config.IBMQ_API_TOKEN
            )
            # 2. ON RÉCUPÈRE L'OBJET BACKEND RÉEL (pas juste son nom)
            real_backend = service.backend(config.HARDWARE_BACKEND_NAME)
            
            # 3. ON PASSE L'OBJET À PENNYLANE
            self.dev = qml.device(
                config.HARDWARE_DEVICE, # "qiskit.remote"
                wires=self.n_qubits_batch,
                backend=real_backend, # <-- On passe l'objet, pas le string !
                shots=config.HARDWARE_SHOTS
            )
            print(f"Connecté à {real_backend.name}. Batching activé.")
        except Exception as e:
            print(f"Erreur de connexion IBMQ : {e}")
            print("Astuce : Vérifiez votre token dans .env ou le nom du backend.")
            raise e

    def get_oracle_for_batch(self, batch_vectors, query_vector, num_to_mark=1):
        """
        Oracle adapté pour un lot spécifique.
        """
        # Top-K classique "Tricheur" (Hybrid Oracle) sur le petit lot uniquement
        similarities = real_np.dot(batch_vectors, query_vector)
        
        # On marque seulement le meilleur du lot pour simplifier le circuit
        # (Plus on marque d'éléments, plus l'algo est complexe)
        best_idx_local = real_np.argmax(similarities)
        
        # Inversion de phase sur l'index local (0 à batch_size-1)
        qml.FlipSign(int(best_idx_local), wires=range(self.n_qubits_batch))

    def get_diffusion(self):
        """Diffuseur standard sur les qubits du batch."""
        for w in range(self.n_qubits_batch): 
            qml.Hadamard(wires=w)
        qml.FlipSign(0, wires=range(self.n_qubits_batch))
        for w in range(self.n_qubits_batch): 
            qml.Hadamard(wires=w)

    def process_batch(self, batch_vectors, query_vector):
        """
        Exécute Grover sur un seul lot de documents.
        """
        # Calcul dynamique des itérations pour ce lot
        # N = taille du lot actuel (peut être < batch_size pour le dernier)
        current_batch_len = len(batch_vectors)
        
        # Si le lot est trop petit (ex: reste 1 doc), pas besoin de quantique
        if current_batch_len == 1:
            return [1.0] # Probabilité 100%

        # Nombre d'itérations Grover optimal pour ce lot
        n_iterations = int(np.pi / 4 * np.sqrt(current_batch_len))
        if n_iterations < 1: n_iterations = 1
        if n_iterations > 3: n_iterations = 3 # Hard-limit pour réduire le bruit sur IBM

        @qml.qnode(self.dev)
        def circuit():
            # Superposition
            for i in range(self.n_qubits_batch):
                qml.Hadamard(wires=i)
            
            # Grover Loop
            for _ in range(n_iterations):
                self.get_oracle_for_batch(batch_vectors, query_vector, num_to_mark=1)
                self.get_diffusion()
            
            return qml.probs(wires=range(self.n_qubits_batch))

        # Exécution sur IBM Q (C'est ici que ça prend du temps !)
        try:
            probs = circuit()
            return probs
        except Exception as e:
            print(f"Erreur sur le batch (skip) : {e}")
            return real_np.zeros(self.batch_size)

    def search(self, query_text, pca_model, embedding_model, top_k=3):
        print(f"Démarrage de la recherche Hardware ({self.total_docs} docs)...")
        
        # 1. Prép. Requête
        q_emb = embedding_model.encode([query_text])
        q_red = pca_model.transform(q_emb)
        q_norm = q_red / (real_np.linalg.norm(q_red) + 1e-10)
        query_vector = q_norm.flatten()
        
        all_candidates = []
        
        # 2. Boucle sur les Batchs
        # On découpe les documents en tranches de 'batch_size'
        total_batches = math.ceil(self.total_docs / self.batch_size)
        
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.total_docs)
            
            batch_vecs = self.vectors[start_idx:end_idx]
            
            print(f"  > Batch {i+1}/{total_batches} (Docs {start_idx}-{end_idx})...")
            
            # Appel Hardware
            # Attention : Padding si le dernier batch est plus petit que n_qubits_batch
            # Pour simplifier ici, on assume que process_batch gère le retour
            batch_probs = self.process_batch(batch_vecs, query_vector)
            
            # Récupération du "Gagnant" local de ce batch
            # On ne garde que le meilleur candidat de chaque batch pour économiser la mémoire/temps
            best_local_idx = real_np.argmax(batch_probs[:len(batch_vecs)])
            best_global_idx = start_idx + best_local_idx
            score = float(batch_probs[best_local_idx])
            
            all_candidates.append({
                "index": int(best_global_idx),
                "probability": score,
                "text": self.chunks[best_global_idx],
                "vector": self.vectors[best_global_idx]
            })
            
            # Pause pour ne pas spammer la queue IBM (optionnel mais poli)
            time.sleep(1)

        # 3. Agrégation et Tri Final
        # On trie les gagnants de chaque batch par leur probabilité quantique
        sorted_candidates = sorted(all_candidates, key=lambda x: x['probability'], reverse=True)
        
        print("Recherche Hardware terminée.")
        return sorted_candidates[:top_k], 0 # Le temps n'est pas pertinent ici (trop long)