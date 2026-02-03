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

# Paramètres de la pipeline quantique simulation
SIMULATION_DEVICE = "lightning.qubit"

HARDWARE_DEVICE = "qiskit.remote" # Backend Qiskit pour l'exécution sur matériel IBM Quantum réel
# Paramètres pour l'exécution sur du matériel quantique réel (IBM Quantum)
IBMQ_API_TOKEN = os.getenv("IBMQ_API_TOKEN") # Clé API IBM Quantum pour l'accès aux ordinateurs quantiques réels
# 'ibm_brisbane' est un processeur Eagle de 127 qubits souvent disponible.
# Autres options possibles : 'ibm_kyoto', 'ibm_osaka', 'ibm_sherbrooke'.
# On peut aussi mettre 'ibmq_qasm_simulator' pour tester la chaîne Qiskit sans attente.
HARDWARE_BACKEND_NAME = "ibm_fez"
# Nombre de qubits physiques disponibles sur la machine choisie (ex: 127 pour Brisbane)
# Sert à calibrer le batching (découpage des données).
HARDWARE_MAX_QUBITS = 156
# Niveau d'optimisation du transpilateur Qiskit (3 = le plus agressif pour réduire la profondeur)
TRANSPILATION_LEVEL = 3
# Nombre de "shots" (tirs) pour l'échantillonnage statistique sur hardware réel
# Plus ce nombre est élevé, plus la distribution de probabilité est précise (mais c'est plus lent).
HARDWARE_SHOTS = 4096