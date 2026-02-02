import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pennylane as qml
# Configuration de l'environnement
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"

# Résolution des imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_pipeline.simulation.grover_retriever import QuantumGroverRetriever
from sentence_transformers import SentenceTransformer

def run_visualizer(query="Effet tunnel quantique"):
    print(f"\n--- VISUALISATION GROVER : {query} ---")
    
    # 1. Chargement des données
    with open("data/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    retriever = QuantumGroverRetriever() # Backend lightning qubits
    
    # 2. Préparation du Vecteur Requête
    # Pipeline identique à l'ingestion : Embedding -> PCA -> Normalisation
    q_emb = model.encode([query])
    q_red = data["pca"].transform(q_emb)
    q_norm = q_red / (np.linalg.norm(q_red) + 1e-10)
    q_vec = q_norm.flatten()

    # 3. Calcul de la "Vérité Terrain" (Cibles Classiques)
    # On détermine mathématiquement quels indices l'Oracle est censé marquer
    # pour pouvoir comparer les pics quantiques avec la réalité.
    sims = np.dot(data["vectors"], q_vec)
    target_indices = np.argpartition(sims, -5)[-5:]
    print(f"Indices cibles (Classique) : {target_indices}")

    # 4. Simulation du Circuit Quantique
    # On utilise 8 itérations (Safe Mode) pour être sûr d'observer la phase montante 
    # de l'amplification sans risquer l'over-rotation (redescente de la probabilité).
    n_iters = 8 
    print(f"Simulation de {n_iters} itérations (Safe Mode)...")

    @qml.qnode(retriever.dev)
    def circuit():
        # A. Superposition uniforme (État |+>)
        for i in range(retriever.n_qubits): qml.Hadamard(wires=i)
        
       # B. Boucle de Grover
        for _ in range(n_iters):
            retriever.get_oracle(q_vec, num_to_mark=5)
            retriever.get_diffusion()
            
        return qml.probs(wires=range(retriever.n_qubits))

    probs = circuit()

    # 5. Diagnostic Chiffré
    # Analyse de l'efficacité de l'amplification (Signal sur Bruit)
    max_prob = np.max(probs)
    max_idx = np.argmax(probs)
    avg_prob = np.mean(probs)
    
    print(f"\nRÉSULTATS STATISTIQUES :")
    print(f"   > Probabilité Max atteinte : {max_prob:.6f}")
    print(f"   > Index du Max : {max_idx}")
    print(f"   > Probabilité Moyenne (Bruit) : {avg_prob:.6f}")
    print(f"   > Facteur d'Amplification : x{max_prob/avg_prob:.2f}")

    if max_prob < avg_prob * 2:
        print("ATTENTION : Amplification faible détectée.")
    else:
        print("SUCCÈS : Amplification confirmée !")

    # 6. GRAPHIQUE (Avec échelle forcée)
    plt.figure(figsize=(12, 6))
    
    # On trace tout en gris clair
    plt.bar(range(len(probs)), probs, color='#bdc3c7', width=1.0, label='Bruit de fond')
    
    # On recolorie les cibles en BLEU VIF
    for t_idx in target_indices:
        plt.bar(t_idx, probs[t_idx], color='#2980b9', width=8.0, label='Cible Oracle' if t_idx==target_indices[0] else "")
    
    # On trace les lignes rouges classiques
    for i, idx in enumerate(target_indices):
        plt.axvline(x=idx, color='#e74c3c', linestyle='--', alpha=0.7)

    plt.title(f"Signature Quantique de Grover (Iters={n_iters})\nMax Prob: {max_prob:.4f} (x{max_prob/avg_prob:.1f} vs Bruit)")
    plt.xlabel("Index Document")
    plt.ylabel("Probabilité")
    plt.legend()
    
    # ZOOM AUTOMATIQUE : On coupe le haut du graphique pour bien voir les pics
    # On laisse 20% de marge au dessus du pic max
    plt.ylim(0, max_prob * 1.2) 
    
    output = "data/grover_final_check.png"
    plt.savefig(output)
    print(f"\nGraphique sauvegardé : {output}")
    print("Ouvrez ce fichier pour voir la preuve !")
    plt.show()

if __name__ == "__main__":
    run_visualizer()