import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
import pennylane as qml
from pennylane import numpy as np
import numpy as real_np # Numpy classique pour les manipulations de listes
import pickle
import config
from qiskit_ibm_runtime import QiskitRuntimeService

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
        # Si aucun ou un seul candidat (cas limites), on retourne les retourne directement
        if num_nodes == 0: return []
        if num_nodes == 1: return candidates
        
        candidate_vecs = np.array([c['vector'] for c in candidates])
        
        # --- PRÉPARATION CLASSIQUE ---
        # Calcul des matrices de poids pour définir le problème physique
        relevance = np.dot(candidate_vecs, query_vec) # Score de similarité cosinus
        redundancy = np.dot(candidate_vecs, candidate_vecs.T) # Matrice de similarité inter-documents

        # Construction des Hamiltoniens
        H_cost = self.build_hamiltonian(relevance, redundancy)
        # Hamiltonien de mélange (Mixer) : Permet de passer d'un état à l'autre via des rotations X
        H_mixer = qml.Hamiltonian([1.0] * num_nodes, [qml.PauliX(i) for i in range(num_nodes)])

        try:
            # 1. CONNEXION MANUELLE
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=config.IBMQ_API_TOKEN
            )
            # 2. RÉCUPÉRATION DE L'OBJET
            real_backend = service.backend(config.HARDWARE_BACKEND_NAME)
            
            # 3. CRÉATION DU DEVICE
            dev = qml.device(
                config.HARDWARE_DEVICE, # "qiskit.remote"
                wires=num_nodes,
                backend=real_backend, # <-- L'objet ici
                shots=config.HARDWARE_SHOTS,
                # On enlève 'ibm_auth_token' ici car le service est déjà authentifié
                options={"resilience_level": 1, "optimization_level": 3}
            )
        except Exception as e:
            print(f"Erreur connexion IBMQ (QAOA) : {e}")
            # Fallback sur simulateur local si le réseau plante
            dev = qml.device("default.qubit", wires=num_nodes)

        # Fonction commune pour construire les portes (évite la duplication de code)
        def qaoa_layer(params):
            # Initialisation |+>
            for i in range(num_nodes):
                qml.Hadamard(wires=i)
            
            # Couches QAOA
            gammas = params[:n_layers]
            betas = params[n_layers:]
            for i in range(n_layers):
                qml.ApproxTimeEvolution(H_cost, gammas[i], 1)
                qml.ApproxTimeEvolution(H_mixer, betas[i], 1)

        # 1. Circuit de Coût (Pour l'optimiseur SPSA)
        # Retourne l'espérance de l'énergie (Valeur réelle à minimiser)
        @qml.qnode(dev)
        def cost_circuit(params):
            qaoa_layer(params)
            return qml.expval(H_cost)

        # 2. Circuit de Probabilité (Pour le résultat final)
        # Retourne la distribution des états
        @qml.qnode(dev)
        def prob_circuit(params):
            qaoa_layer(params)
            return qml.probs(wires=range(num_nodes))

        # --- EXÉCUTION QUANTIQUE ---
        
        params = np.array([0.1] * (2 * n_layers), requires_grad=True)
        
        # Boucle d'optimisation SPSA (Stochastic Parameter Shift Algorithm)
        # pour trouver les meilleurs paramètres gamma/beta.
        # Choix basé sur le papier scientifique.
        opt = qml.SPSAOptimizer(maxiter=10, c=0.15, a=0.2)
        print(f"  > Démarrage boucle SPSA sur {config.HARDWARE_BACKEND_NAME} (5 itérations)...")
        print("    (Cela peut prendre plusieurs minutes selon la file d'attente...)")
        try:
            # La boucle tourne sur le hardware !
            for i in range(5):
                params, cost = opt.step_and_cost(cost_circuit, params)
                print(f"    - Step {i+1}/5 terminés.")
        except Exception as e:
            print(f"⚠️ Timeout ou Erreur IBMQ pendant l'optimisation : {e}")
            print("    -> Utilisation des paramètres courants.")
        
        # --- MESURE FINALE ---
        print("  > Mesure finale sur puce...")
        try:
            probs = prob_circuit(params)
        except Exception as e:
            print(f"❌ Erreur finale QAOA : {e}")
            # En cas de crash total du hardware, on renvoie une proba uniforme simulée
            probs = real_np.ones(2**num_nodes) / (2**num_nodes)

        # --- POST-TRAITEMENT ---
        best_bitstring_idx = real_np.argmax(probs)
        bitstring = format(best_bitstring_idx, f'0{num_nodes}b')
        selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
        
        # Fallback Robuste (Top 3 classique si échec)
        if not selected_indices: 
            best_indices_classic = real_np.argsort(relevance)[::-1]
            k_rescue = min(3, num_nodes)
            selected_indices = best_indices_classic[:k_rescue].tolist()
            
        return [candidates[i] for i in selected_indices]