from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any
from langchain_openai import ChatOpenAI
import numpy as np

# Imports des modules classiques et de la config partagée
from classic_pipeline.retriever import ClassicalRetriever
from classic_pipeline.ranker import ClassicalRanker
from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, LAMBDA_DIVERSITY

# 1. Définition de l'état (Mémoire du pipeline classique)
class ClassicalAgentState(TypedDict):
    query: str
    pca: Any
    embed_model: Any
    vectors: Any # Base vectorielle complète (numpy array)
    chunks: Any # Textes bruts
    candidates: List[dict] # Résultats bruts de la recherche vectorielle
    final_docs: List[dict] # Résultats re-classés par MMR
    answer: str
    
# 2. Les Nœuds (Étapes du Pipeline)
def classical_retrieval_node(state: ClassicalAgentState):
    """Étape 1 : Recherche vectorielle standard (Cosine Similarity)."""
    print("\n--- [NODE] CLASSICAL RETRIEVAL (COSINE) ---")
    
    # Instanciation à la volée du retriever classique
    retriever = ClassicalRetriever(state["vectors"], state["chunks"])
    
    # Recherche top-k=8 (Même fenêtre que pour le quantique pour comparaison équitable)
    docs = retriever.search(
        state["query"], 
        state["pca"], 
        state["embed_model"], 
        top_k=8
    )
    return {"candidates": docs}

def classical_rank_node(state: ClassicalAgentState):
    """Étape 2 : Re-ranking classique via MMR pour la diversité."""
    print("--- [NODE] CLASSICAL RANKING (MMR) ---")
    
    # Utilisation du même lambda que QAOA pour comparer "pommes vs pommes"
    ranker = ClassicalRanker(lambda_div=LAMBDA_DIVERSITY)
    
    # Sélection des 3 meilleurs documents diversifiés
    ranked_docs = ranker.rank(state["candidates"], top_n=3)
    return {"final_docs": ranked_docs}

def generate_answer_node(state: ClassicalAgentState):
    """Étape 3 : Génération de la réponse (Prompt identique au flux quantique)."""
    print(f"--- [NODE] LLM GENERATION ({LLM_MODEL}) ---")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "QRAG Classical Baseline",
        }
    )
    # Construction du contexte
    context = "\n\n".join([f"Source {i+1}: {d['text']}" for i, d in enumerate(state["final_docs"])])
    # Prompt strictement identique à celui du pipeline quantique pour isoler l'impact du Retrieval
    prompt = f"""Tu es un expert en informatique quantique. Réponds à la question suivante en utilisant UNIQUEMENT le contexte fourni. 
    Si l'information n'est pas dans le contexte, dis que tu ne sais pas.

    Question: {state['query']}
    
    Contexte:
    {context}
    
    Réponse (en français) :"""
    
    return {"answer": llm.invoke(prompt).content}

# 3. Construction du Graphe
def create_classical_graph():
    workflow = StateGraph(ClassicalAgentState)
    
    # Ajout des nœuds
    workflow.add_node("retrieve", classical_retrieval_node)
    workflow.add_node("rank", classical_rank_node)
    workflow.add_node("generate", generate_answer_node)
    
    # Définition du flux linéaire : Retrieval -> Ranking -> LLM -> Fin
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rank")
    workflow.add_edge("rank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()