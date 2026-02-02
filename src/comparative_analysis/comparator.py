import os
import pickle
from config import PCA_DIMENSIONS, LLM_MODEL
from classic_pipeline.bruteforce.workflow_bruteforce import create_classical_graph
from quantum_pipeline.simulation.workflow import create_qrag_graph
from quantum_pipeline.simulation.grover_retriever import QuantumGroverRetriever
from sentence_transformers import SentenceTransformer

class QRAGComparator:
    """
    Orchestrateur du duel : Lance les deux pipelines (Quantique vs Classique) en parallèle
    sur la même requête et affiche les résultats côte-à-côte pour analyse qualitative.
    """
    def __init__(self):
        print(f"--- Initialisation du Comparateur (Espace {PCA_DIMENSIONS}D) ---")
        
        # 1. Chargement centralisé des données (Optimisation RAM)
        # On charge le dataset une seule fois en mémoire pour le partager entre les deux agents.
        with open("data/processed_data.pkl", "rb") as f:
            self.data = pickle.load(f)
        
        # 2. Initialisation des modèles partagés
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.retriever = QuantumGroverRetriever()
        
        # 3. Compilation des Graphes LangGraph
        self.q_graph = create_qrag_graph() # Pipeline Grover + QAOA
        self.c_graph = create_classical_graph() # Pipeline Cosine + MMR

    def run_duel(self, query):
        print(f"\nDuel lancé pour la question : '{query}'")
        
       # --- A. Exécution Pipeline Quantique ---
        q_inputs = {
            "query": query, "pca": self.data["pca"], "embed_model": self.model,
            "retriever": self.retriever, "candidates": [], "final_docs": [], "answer": ""
        }
        # invoke() lance le graphe défini dans quantum_pipeline/workflow.py
        q_res = self.q_graph.invoke(q_inputs)

        # --- B. Exécution Pipeline Classique ---
        c_inputs = {
            "query": query, "pca": self.data["pca"], "embed_model": self.model,
            "vectors": self.data["vectors"], "chunks": self.data["chunks"],
            "candidates": [], "final_docs": [], "answer": ""
        }
        # invoke() lance le graphe défini dans classic_pipeline/workflow.py
        c_res = self.c_graph.invoke(c_inputs)

        # --- C. Affichage Comparatif ---
        print("\n" + "="*80)
        print(f"RÉSULTATS DU DUEL (LLM: {LLM_MODEL})")
        print("="*80)
        
        print(f"\n[QRAG QUANTIQUE]\n{q_res['answer']}")
        print("\nSources utilisées par Grover + QAOA :")
        for d in q_res['final_docs']: print(f" - {d['text'][:70]}...")

        print("\n" + "-"*40)
        
        print(f"\n[RAG CLASSIQUE]\n{c_res['answer']}")
        print("\nSources utilisées par Cosine + MMR :")
        for d in c_res['final_docs']: print(f" - {d['text'][:70]}...")
        print("="*80)

if __name__ == "__main__":
    comparator = QRAGComparator()
    comparator.run_duel("Donne moi la définition de l\'effet tunnel quantique.")