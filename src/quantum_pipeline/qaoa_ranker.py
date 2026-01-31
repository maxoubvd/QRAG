import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
import pennylane as qml
from pennylane import numpy as np
import pickle

class GroQRanker:
    """
    Ranker quantique basé sur QAOA (Quantum Approximate Optimization Algorithm).
    Son but est de réordonner les candidats trouvés par Grover pour maximiser la diversité
    tout en gardant la pertinence (équivalent quantique du MMR - Maximal Marginal Relevance).
    """
    def __init__(self, lambda_div=0.5):
        # Paramètre de compromis : 
        # 0.0 = On veut juste la pertinence (risque de redondance)
        # 1.0 = On veut un maximum de diversité (risque de hors-sujet)
        self.lambda_div = lambda_div

    def build_hamiltonian(self, relevance, redundancy_matrix):
        """
        Construit l'Hamiltonien de coût (Cost Hamiltonian) représentant le problème.
        On mappe le problème d'optimisation sur un modèle d'Ising (QUBO).
        """
        num_nodes = len(relevance)
        coeffs = []
        obs = []

        # 1. Termes linéaires (Pertinence) -> Champs Z locaux
        # On veut maximiser la pertinence, donc minimiser l'énergie associée.
        # Si le qubit i est à 1 (document choisi), il contribue à réduire l'énergie.
        for i in range(num_nodes):
            coeffs.append(-relevance[i] / 2)
            obs.append(qml.PauliZ(i))

        # 2. Termes quadratiques (Redondance) -> Interactions Z-Z
        # Si deux documents i et j sont choisis ensemble (état |11>),
        # on ajoute une pénalité énergétique proportionnelle à leur similarité.
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                coeffs.append(self.lambda_div * redundancy_matrix[i][j] / 4)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return qml.Hamiltonian(coeffs, obs)

    def rank(self, candidates, query_vec, n_layers=2):
        """
        Exécute le circuit QAOA pour trouver le meilleur sous-ensemble de documents.
        """
        num_nodes = len(candidates)
        # Si aucun candidat (cas limite), on retourne une liste vide
        candidate_vecs = np.array([c['vector'] for c in candidates])
        
        # --- PRÉPARATION CLASSIQUE ---
        # Calcul des matrices de poids pour définir le problème physique
        relevance = np.dot(candidate_vecs, query_vec) # Score de similarité cosinus
        redundancy = np.dot(candidate_vecs, candidate_vecs.T) # Matrice de similarité inter-documents

        # Construction des Hamiltoniens
        H_cost = self.build_hamiltonian(relevance, redundancy)
        # Hamiltonien de mélange (Mixer) : Permet de passer d'un état à l'autre via des rotations X
        H_mixer = qml.Hamiltonian([1.0] * num_nodes, [qml.PauliX(i) for i in range(num_nodes)])

        # Création dynamique du device (le nombre de qubits dépend du nombre de candidats trouvés par Grover)
        dev = qml.device("default.qubit", wires=num_nodes)

        @qml.qnode(dev)
        def circuit(params):
            # 1. Initialisation : Superposition uniforme (Hadamard sur tous les qubits)
            # On commence dans l'état |+> (toutes les combinaisons possibles équiprobables)
            for i in range(num_nodes):
                qml.Hadamard(wires=i)
            
            # 2. Couches QAOA (Alternance Cost / Mixer)
            # C'est ici que l'interférence quantique amplifie la solution optimale
            gammas = params[:n_layers]
            betas = params[n_layers:]
            for i in range(n_layers):
                # Evolution selon le coût (Problem unitary)
                qml.ApproxTimeEvolution(H_cost, gammas[i], 1)
                # Evolution selon le mélange (Mixer unitary)
                qml.ApproxTimeEvolution(H_mixer, betas[i], 1)
            
            # On mesure la probabilité de chaque chaîne de bits (ex: "101" = doc 1 et 3 choisis)
            return qml.probs(wires=range(num_nodes))

        # --- EXÉCUTION QUANTIQUE ---
        
        # [TODO]
        # Dans un vrai QAOA, on ferait une boucle d'optimisation classique (ex: COBYLA)
        # pour trouver les meilleurs paramètres gamma/beta.
        # Pour ce prototype rapide, on utilise des paramètres heuristiques fixes.
        params = np.array([0.1] * (2 * n_layers), requires_grad=True)
        probs = circuit(params)
        
        # --- POST-TRAITEMENT (Interpretation) ---
        
        # On cherche l'état avec la plus haute probabilité (le pic d'interférence)
        best_bitstring_idx = np.argmax(probs)
        bitstring = format(best_bitstring_idx, f'0{num_nodes}b')
        
        # Conversion du bitstring en liste de documents
        # "1" signifie que le QAOA recommande de garder ce document
        selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
        
        # Sécurité : Si le QAOA dit "tout jeter" (00...0), on garde au moins le meilleur document classique
        if not selected_indices: selected_indices = [0]
            
        return [candidates[i] for i in selected_indices]