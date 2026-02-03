import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Coupe les logs TensorFlow
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Imports des modules locaux (Config, Pipeline Quantique, Pipeline Classique)
import config
from quantum_pipeline.simulation.grover_retriever_sim import QuantumGroverRetriever
from quantum_pipeline.simulation.grover_retriever_sim_noise import QuantumGroverRetrieverNoise
from quantum_pipeline.simulation.quantum_workflow_sim import create_qrag_graph
from classic_pipeline.bruteforce.classical_workflow_bruteforce import create_classical_graph


load_dotenv() # Chargement des variables d'environnement depuis le fichier .env

class QRAGDuel:
    def __init__(self, use_noise=False, noise_level=0.05):
        print(f"Initialisation du système (Mode {config.PCA_DIMENSIONS}D)...")
        # Chargement du dataset vectorisé et des objets PCA pré-calculés
        with open("data/processed_data.pkl", "rb") as f:
            self.data = pickle.load(f)
        
        self.pca = self.data["pca"]
        self.model = SentenceTransformer(config.EMBED_MODEL_NAME) # Modèle d'embedding partagé
        if use_noise:
            self.retriever = QuantumGroverRetrieverNoise(noise_level=noise_level)
        else:
            self.retriever = QuantumGroverRetriever() # Mode Idéal
        
        # Initialisation des graphes LangGraph
        self.quantum_graph = create_qrag_graph()
        self.classical_graph = create_classical_graph()

    def run(self, query):
        print(f"\nREQUÊTE : {query}")
        
        # 1. EXECUTION QUANTIQUE
        print("\n" + "="*25 + " FLUX QUANTIQUE (Grover + QAOA) " + "="*25)
        # Préparation des inputs pour le graphe quantique
        q_inputs = {
            "query": query, "pca": self.pca, "embed_model": self.model,
            "retriever": self.retriever, "candidates": [], "final_docs": [], "answer": ""
        }
        q_result = self.quantum_graph.invoke(q_inputs)

        # 2. EXECUTION CLASSIQUE
        print("\n" + "="*25 + " FLUX CLASSIQUE (Cosine + MMR) " + "="*25)
        # Préparation des inputs pour le graphe classique (Baseline de comparaison)
        c_inputs = {
            "query": query, "pca": self.pca, "embed_model": self.model,
            "vectors": self.data["vectors"], "chunks": self.data["chunks"],
            "candidates": [], "final_docs": [], "answer": ""
        }
        c_result = self.classical_graph.invoke(c_inputs)

        # 3. AFFICHAGE DU COMPARATIF
        print("\n" + "-"*80)
        print(f"RÉSULTATS DU DUEL")
        print("-" * 80)
        print(f"QUANTIQUE :\n{q_result['answer']}")
        print("-" * 80)
        print(f"CLASSIQUE :\n{c_result['answer']}")
        print("-"*80)

if __name__ == "__main__":
# Vérification de l'existence des données avant démarrage
    if not os.path.exists("data/processed_data.pkl"):
        print("Erreur : data/processed_data.pkl introuvable. Lancez 'python src/data_prep.py' d'abord.")
    else:
        print("\n--- CONFIGURATION DE LA SIMULATION ---")
        print("1. Simulation IDÉALE (Pas de bruit, résultat théorique parfait)")
        print("2. Simulation BRUITÉE (Simulation NISQ, erreurs de décohérence)")
        
        choice = input("Votre choix (1 ou 2) : ").strip()
        
        use_noise = False
        noise_level = 0.0
        
        if choice == "2":
            use_noise = True
            try:
                lvl = input("Entrez le niveau de bruit (ex: 0.05 pour 5%) [Défaut: 0.05] : ").strip()
                noise_level = float(lvl) if lvl else 0.05
            except ValueError:
                print("Valeur invalide, utilisation de 0.05 par défaut.")
                noise_level = 0.05
        
        # Lancement avec les paramètres choisis
        duel = QRAGDuel(use_noise=use_noise, noise_level=noise_level)
        
        query = "Give me the definition of a quantum computer."
        duel.run(query)