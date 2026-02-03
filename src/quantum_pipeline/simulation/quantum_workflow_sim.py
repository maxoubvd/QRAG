import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import numpy as np
# Importation de la configuration centralisée et des modules quantiques
from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, LAMBDA_DIVERSITY
from quantum_pipeline.simulation.grover_retriever_sim import QuantumGroverRetriever
from quantum_pipeline.simulation.grover_retriever_sim_noise import QuantumGroverRetrieverNoise
from quantum_pipeline.simulation.qaoa_ranker_sim import GroQRanker

# 1. Définition de l'état (Mémoire partagée de l'agent)
class AgentState(TypedDict):
    query: str
    pca: Any
    embed_model: Any
    retriever: Any
    candidates: List[dict]
    final_docs: List[dict]
    answer: str

# 2. Les Nœuds (Étapes du Pipeline)
def quantum_retrieval_node(state: AgentState):
    """Étape 1 : Recherche dense amplifiée par Grover."""
    print("\n--- [NODE] QUANTUM RETRIEVAL (GROVER) ---")
    
    # Appel du moteur de recherche quantique
    docs, _ = state["retriever"].search(
        state["query"], 
        state["pca"], 
        state["embed_model"], 
        top_k=8 # On récupère un pool large pour laisser le choix au QAOA ensuite
    )
    return {"candidates": docs}

def groq_rank_node(state: AgentState):
    """Étape 2 : Optimisation combinatoire (QAOA) pour la diversité."""
    print("--- [NODE] QUANTUM RANKING (QAOA) ---")
    
    # Initialisation du Ranker avec le facteur de diversité de la config
    ranker = GroQRanker(lambda_div=LAMBDA_DIVERSITY)
    
    # Préparation du vecteur requête (Embedding -> PCA -> Normalisation)
    # La normalisation est cruciale pour la construction correcte de l'Hamiltonien
    q_emb = state["embed_model"].encode([state["query"]])
    query_vec = state["pca"].transform(q_emb)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    query_vec = query_vec.flatten()

    # Exécution du circuit QAOA pour le re-classement
    ranked = ranker.rank(state["candidates"], query_vec)
    return {"final_docs": ranked}

def generate_answer_node(state: AgentState):
    """Étape 3 : Génération de la réponse via LLM."""
    print(f"--- [NODE] LLM GENERATION ({LLM_MODEL}) ---")
    
    # Configuration du client compatible OpenAI (pour OpenRouter)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "QRAG Quantum Project",
        }
    )
    
    # Construction du contexte à partir des documents validés par QAOA
    context = "\n\n".join([f"Source {i+1}: {d['text']}" for i, d in enumerate(state["final_docs"])])
    # Prompt restrictif pour forcer l'usage du contexte (Anti-Hallucination)
    prompt = f"""TYou are an expert in quantum computing. Answer the following question using ONLY the provided context.
    If the information is not in the context, say you don't know.

    Question: {state['query']}
    
    Context:
    {context}
    
    Answer (in English):"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# 3. Construction du Graphe (Orchestration)
def create_qrag_graph():
    workflow = StateGraph(AgentState)
    
    # Ajout des nœuds
    workflow.add_node("retrieve", quantum_retrieval_node)
    workflow.add_node("rank", groq_rank_node)
    workflow.add_node("generate", generate_answer_node)

    # Définition du flux linéaire : Grover -> QAOA -> LLM -> Fin
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rank")
    workflow.add_edge("rank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()