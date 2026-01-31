import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dotenv import load_dotenv

load_dotenv() # Chargement des variables d'environnement depuis le fichier .env

EMBED_MODEL_NAME = 'all-MiniLM-L6-v2' # Modèle d'embedding utilisé pour vectoriser les documents.

PCA_DIMENSIONS = 64 # 64 dimensions est un compromis optimal pour la simulation quantique :
# - Suffisamment riche pour garder le sens sémantique.
# - Suffisamment petit pour être encodé sur ~6 à 12 qubits sans faire exploser le CPU.
LAMBDA_DIVERSITY = 0.8 # Facteur de diversité pour le re-classement (MMR / QAOA).
# - 1.0 = Pure pertinence (risque de redondance).
# - 0.0 = Pure diversité (risque de hors-sujet).
# 0.8 favorise la pertinence tout en évitant les doublons exacts.

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Clé API pour OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1" # URL de base pour OpenRouter
LLM_MODEL = "tngtech/deepseek-r1t2-chimera:free" # Modèle LLM utilisé pour la génération de réponses (gratuit).