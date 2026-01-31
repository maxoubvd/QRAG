import os
import sys
import time
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PCA_DIMENSIONS
from quantum_pipeline.grover_retriever import QuantumGroverRetriever
from sentence_transformers import SentenceTransformer

def run_performance_benchmark(n_queries=10):
    """
    Compare la précision et la vitesse entre la recherche Classique et Quantique (Simulée).
    n_queries est réduit par défaut car la simulation matricielle est lourde pour le CPU.
    """
    # Chargement des données pré-calculées
    data_path = os.path.join(os.getcwd(), "data", "processed_data.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    vectors = data["vectors"]
    chunks = data["chunks"]
    pca = data["pca"]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    retriever = QuantumGroverRetriever() # Initialise le simulateur (Lightning Qubit)
    
    n_docs = len(vectors)
    print(f"\n--- Benchmark Performance : Quantique vs Classique ---")
    print(f"Docs: {n_docs} | Qubits: {retriever.n_qubits} | Tests: {n_queries}")
    
    # Sélection aléatoire de documents qui serviront de "requêtes"
    # Cela garantit qu'il existe une réponse parfaite dans la base (le document lui-même).
    np.random.seed(42)
    indices = np.random.choice(n_docs, n_queries, replace=False)
    
    stats = {"class_lat": [], "quant_lat": [], "overlap": []}

    print("\nLancement du duel (Patience, simulation en cours)...")
    
    for i, idx in enumerate(indices):
        print(f"   > Test {i+1}/{n_queries}...", end="\r") # Barre de progression
        
        # On utilise les 100 premiers caractères d'un document comme requête
        query_text = chunks[idx][:100] 
        
        # --- 1. Approche Classique (Baseline) ---
        # Calcul vectoriel direct (Produit scalaire). Très rapide (ms).
        # Sert de "Vérité Terrain".
        start = time.perf_counter()
        q_red = pca.transform(model.encode([query_text]))
        q_norm = q_red / (np.linalg.norm(q_red) + 1e-10)
        scores = np.dot(vectors, q_norm.T).flatten()
        top_k_class = np.argsort(scores)[::-1][:5]
        stats["class_lat"].append(time.perf_counter() - start)

        # --- 2. Approche Quantique (Grover) ---
        # Simulation du circuit quantique.
        # Le temps inclut la simulation mathématique lourde des 12 qubits.
        res_quant, duration = retriever.search(query_text, pca, model, top_k=5)
        top_k_quant = [r['index'] for r in res_quant]
        stats["quant_lat"].append(duration)

        # --- 3. Mesure de Fidélité (Overlap) ---
        # On regarde combien de documents trouvés par le Classique sont aussi trouvés par Grover.
        # 1.0 (100%) = Le quantique fonctionne mathématiquement parfaitement.
        common = set(top_k_class) & set(top_k_quant)
        common = set(top_k_class) & set(top_k_quant)
        stats["overlap"].append(len(common) / 5)

    print(f"   > Test {n_queries}/{n_queries} Terminé !       ")
    
    # --- RÉSULTATS ---
    print("\n" + "RÉSULTATS DU BENCHMARK")
    print(f"------------------------------------")
    print(f"Latence Classique (Moy) : {np.mean(stats['class_lat'])*1000:.2f} ms")
    print(f"Latence Quantique (Moy) : {np.mean(stats['quant_lat']):.2f} s  <-- Normal si élevé en simulation")
    print(f"------------------------------------")
    print(f"FIDÉLITÉ QUANTIQUE (OVERLAP) : {np.mean(stats['overlap'])*100:2.1f}%")
    print(f"------------------------------------")
    
    if np.mean(stats['overlap']) > 0.7:
        print("SUCCÈS : Le moteur quantique reproduit fidèlement la recherche sémantique.")
    else:
        print("NOTE : Un overlap faible peut venir du bruit de la PCA 64D.")

if __name__ == "__main__":
    run_performance_benchmark()