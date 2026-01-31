import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Coupe les logs TensorFlow
import json
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import config 

def load_jsonl_data(filepath):
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append({"text": data["text"], "url": data.get("url", "")})
    return documents

print(f"Chargement du modèle d'embedding : {config.EMBED_MODEL_NAME}")
model = SentenceTransformer(config.EMBED_MODEL_NAME)

# 1. Chargement
raw_data = load_jsonl_data("documents/crawler_output.jsonl")

# 2. Chunking
# Découpage du texte en segments de 400 caractères avec chevauchement pour garder le contexte.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = []
for doc in raw_data:
    split_texts = text_splitter.split_text(doc["text"])
    chunks.extend(split_texts)

print(f"Nombre de chunks générés : {len(chunks)}")

# 3. Embeddings & PCA
print("Génération des embeddings...")
embeddings = model.encode(chunks) # Vectorisation initiale (384 dimensions pour all-MiniLM)

print(f"Calcul de la PCA (Dimensions: {config.PCA_DIMENSIONS})...")
# Réduction de dimension (ex: 384 -> 64) pour limiter le nombre de qubits requis (2^n >= dim).
pca = PCA(n_components=config.PCA_DIMENSIONS)
reduced_vecs = pca.fit_transform(embeddings)

# 4. Normalisation L2 (Crucial pour l'Amplitude Encoding)
# Les états quantiques doivent avoir une norme de 1. Cette étape est mathématiquement obligatoire.
norms = np.linalg.norm(reduced_vecs, axis=1, keepdims=True)
reduced_vecs = reduced_vecs / (norms + 1e-10)

# 5. Sauvegarde
if not os.path.exists("data"):
    os.makedirs("data")

# On sauvegarde les chunks, les vecteurs réduits et l'objet PCA (pour transformer les futures requêtes de la même façon).
with open("data/processed_data.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "pca": pca, "vectors": reduced_vecs}, f)

print("Données prêtes pour le duel QRAG vs Classique.")