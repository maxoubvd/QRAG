# QRAG: Quantum-Enhanced Retrieval-Augmented Generation

**QRAG** est un moteur de recherche hybride nouvelle génération qui combine la puissance de l'informatique quantique (Algorithme de Grover et QAOA) avec les modèles de langage modernes (LLMs) pour une recherche d'information ultra-précise.

Contrairement aux RAG classiques basés uniquement sur la similarité cosinus, QRAG utilise l'**interférence quantique** pour amplifier la probabilité des documents les plus pertinents et filtrer mathématiquement le bruit sémantique.

---

## Table des Matières

* [Description](#description)
* [Architecture & Choix Techniques](#architecture--choix-techniques)
* [Algorithmes Quantiques Implémentés](#algorithmes-quantiques-implémentés)
* [Installation](#installation)
* [Utilisation](#utilisation)
* [Benchmarks et Résultats](#benchmarks-et-résultats)
* [Structure du Projet](#structure-du-projet)
* [Roadmap & TODO](#roadmap--todo)

---

## Description

QRAG implémente un pipeline RAG hybride modulaire où :

1. **La Recherche (Retrieval)** utilise l'Algorithme de Grover pour une récupération théorique en $O(n)$ au lieu de $O(\sqrt n)$.
2. **Le Classement (Ranking)** emploie QAOA (Quantum Approximate Optimization Algorithm) pour optimiser la pertinence et la diversité simultanément.
3. **La Génération** utilise des LLMs (via OpenRouter) pour produire des réponses basées sur les documents sélectionnés par le circuit quantique.

Le projet inclut un "Duel" en temps réel entre l'approche quantique et une approche classique (Baseline SOTA) pour mesurer les gains en fidélité.

---

## Architecture & Choix Techniques

Ce projet a été conçu pour comparer scientifiquement l'approche quantique et classique ("Apples-to-Apples").

### 1. Framework : Pourquoi PennyLane ?

Le choix de **Xanadu PennyLane** comme framework quantique, de préférence à Qiskit ou Cirq, repose sur trois piliers stratégiques pour ce projet :

* **Architecture "Quantum Differentiable" (QML Natif)** :
PennyLane est conçu spécifiquement pour l'apprentissage automatique quantique (QML). Contrairement aux SDKs purement physiques, il gère nativement le calcul de gradients à travers les circuits quantiques. C'est indispensable pour notre implémentation du **QAOA**, où les paramètres du circuit doivent être optimisés pour converger vers le meilleur ranking.
* **Performance de Simulation (Lightning)** :
Le plugin `lightning.qubit` offre un backend de simulation d'états écrit en **C++** avec des liaisons (bindings) directes. Cela nous permet de simuler des circuits profonds (Grover 12 qubits + 8 itérations) en moins de 200ms, là où une simulation Python standard prendrait plusieurs secondes.
* **Hardware Agnostique** :
PennyLane agit comme une couche d'abstraction unifiée. Notre code écrit pour le simulateur peut être déployé sur **IBM Quantum** (via `pennylane-qiskit`), **Amazon Braket** ou **Google Quantum AI** en changeant une seule ligne de code (définition du `device`), sans réécrire la logique des algorithmes Grover ou QAOA.

### 2. Stratégie de Simulation vs Hardware Réel

Pour ce prototype, nous avons fait le choix technique de **simuler des qubits parfaits** plutôt que d'utiliser des processeurs quantiques réels (QPU type IBM/Rigetti).

* **Pourquoi ?** L'algorithme de Grover nécessite une grande profondeur de circuit (nombreuses itérations). Les processeurs actuels (NISQ) sont trop bruyants : la décohérence détruit l'information quantique avant la fin du calcul de Grover sur 12 qubits.
* **L'approche :** Nous utilisons `lightning.qubit` (simulateur C++) pour valider la *logique algorithmique* et la *convergence mathématique* sans le bruit matériel, constituant une "Preuve de Concept" prête pour le futur hardware FTQC (Fault-Tolerant Quantum Computing).

### 3. Accélération C++ (`lightning.qubit`)

La simulation de 12+ qubits (pour >2000 documents) implique des calculs matriciels massifs ($2^{12}\ \times 2^{12}$).

* **Choix :** Utilisation du plugin `PennyLane-Lightning`.
* **Impact :** Réduction du temps de recherche de **~15s à <0.2s** par requête, rendant la démo utilisable en temps réel.

### 4. Symétrie des Pipelines

Pour garantir l'intégrité des benchmarks, les deux branches partagent les mêmes pré-traitements :

* **Même Ingestion :** Chunking identique (400 chars).
* **Mêmes Embeddings :** `all-MiniLM-L6-v2` (384 dimensions).
* **Même PCA :** Réduction à 64 dimensions pour nettoyer le bruit et faciliter l'encodage d'amplitude.

---

## Algorithmes Quantiques Implémentés

### 1. Recherche : Algorithme de Grover

Nous utilisons Grover pour amplifier l'amplitude des documents pertinents (Recherche non structurée).

* **Encodage :** Chaque document est mappé sur un état de base $\ket{x}$ parmi $N$ états ($N=2^{12}=4096$).
* **Oracle (`FlipSign`) :** Nous utilisons une opération "Atomique" qui inverse la phase des indices cibles identifiés par le produit scalaire.
* *Action :* $\ket{x} \xrightarrow{Oracle} -\ket{x}$ si  est pertinent.


* **Diffuseur :** Opérateur de réflexion autour de la moyenne. Implémenté via `H -> FlipSign(0) -> H`. Cette implémentation est mathématiquement robuste et évite les erreurs d'arrondi des décompositions de portes complexes.
* **Optimisation des Itérations :** Le nombre d'itérations est calculé dynamiquement ($\approx \frac{\pi}{4}\sqrt{N/M}$) mais plafonnée (Safe Mode) pour éviter l'over-rotation.

### 2. Ranking : QAOA (GroQ-Rank)

Pour trier les résultats de Grover, nous résolvons un problème d'optimisation combinatoire (QUBO) équivalent au MMR (Maximal Marginal Relevance).

* **Objectif :** Sélectionner un sous-ensemble de documents qui maximise la pertinence tout en minimisant la redondance.
* **Hamiltonien de Coût ($H_C$) :**
* *Termes Linéaires ($Z_i$)* : Favorisent la pertinence (Score Cosine).
* *Termes Quadratiques ($Z_i Z_j$)* : Pénalisent la redondance (Similarité inter-doc).


* **Résultat :** Le circuit converge vers un bitstring (ex: `101`) indiquant les meilleurs documents à garder pour le contexte du LLM.

---

## Installation

### Prérequis

* Python 3.8+
* Compilateur C++ (recommandé pour `PennyLane-Lightning`)
* Clé API OpenRouter

### Étapes

1. **Cloner le projet**
```bash
git clone https://github.com/votre-user/QRAG.git
cd QRAG

```


2. **Environnement Virtuel**
```bash
python -m venv qrag_env
source qrag_env/bin/activate  # Windows: .\qrag_env\Scripts\activate

```


3. **Dépendances**
```bash
pip install -r requirements.txt
pip install PennyLane-Lightning  # Crucial pour la vitesse

```


4. **Configuration**
Créez un fichier `.env` à la racine :
```env
OPENROUTER_API_KEY=sk-or-v1-xxxx...

```


5. **Préparation des Données**
```bash
python src/data_prep.py

```



---

## Utilisation

### 1. Le Duel (Démo Principale)

Posez une question et comparez les réponses Quantique vs Classique.

```bash
python src/main.py

```

### 2. Vérification de l'Amplification (Grover)

Générez le graphique prouvant l'effet tunnel et l'amplification de probabilité.

```bash
python src/comparative_analysis/visualizer.py

```

*Le graphique est sauvegardé dans `data/grover_final_check.png`.*

### 3. Benchmark de Performance

Mesurez la fidélité (Overlap) et la latence.

```bash
python src/comparative_analysis/benchmarker.py

```

---

## Benchmarks et Résultats

Tests réalisés sur un corpus de 2136 documents techniques (12 Qubits).

| Métrique | Classique (Baseline) | QRAG (Quantique Simulée) |
| --- | --- | --- |
| **Complexité Algo** | $O(n)$ | $O(\sqrt{n})$ (Théorique) |
| **Fidélité (Overlap)** | Référence | **100%** (Validation parfaite) |
| **Latence (Simulée)** | ~17 ms | ~200 ms (via Lightning) |
| **Amplification Signal** | N/A | **x256.7** vs Bruit de fond |

---

## 📂 Structure du Projet

```
qrag/
├── data/                    # Données vectorisées et graphiques
├── documents/               # Corpus brut (JSONL)
├── src/
│   ├── config.py            # Configuration globale (Protection Env)
│   ├── data_prep.py         # Pipeline d'ingestion (Embed -> PCA)
│   ├── main.py              # Orchestrateur du Duel
│   ├── classic_pipeline/    # Implémentation Baseline (Cosine + MMR)
│   ├── quantum_pipeline/    # Cœur Quantique
│   │   ├── grover_retriever.py # Moteur Grover (Oracle/Diffuseur)
│   │   ├── qaoa_ranker.py      # Moteur QAOA (Optimisation)
│   │   └── workflow.py         # Graphe LangGraph Quantique
│   └── comparative_analysis/
│       ├── benchmarker.py   # Tests de performance
│       ├── visualizer.py    # Preuve graphique
│       └── comparator.py    # Logique du duel
└── requirements.txt

```

---

## Roadmap & TODO

Voici les axes d'amélioration identifiés dans le code actuel (voir les tags `#[TODO]` dans le code) :

* [ ] **Optimisation QAOA (`src/quantum_pipeline/qaoa_ranker.py`)** :
Actuellement, le circuit QAOA utilise des paramètres `gamma/beta` fixes pour garantir la rapidité de la démo.
*TODO : Implémenter une boucle d'optimisation classique (ex: COBYLA) pour calibrer dynamiquement les paramètres du circuit.*
* [ ] **Scalabilité Hardware (Batching)** :
Adapter le `QuantumGroverRetriever` pour découper le dataset en lots (batches) afin de permettre l'exécution sur des machines IBM Quantum réelles (souvent limitées à 127 qubits bruyants).
* [ ] **Modèle de Bruit** :
Ajouter `qml.noise` pour simuler les erreurs de décohérence et tester la robustesse de l'algorithme face à un environnement NISQ réel.
* [ ] Développement d'un frontend de démonstration :
  - Frontend en streamlit.
  - Ecran scindé "split-view": réponse classique vs réponse quantique.
  - Jauge de fidelité quantique et gain d'amplification qui s'anime.
* [ ] Production d'une vidéo de démo
* [ ] Ecriture du rapport  
**Deadline: Ces tâches doivent êtres terminées avant le 15 mars**
* [ ] **Test en condition réelle**:
Test avec un oridnateur Quantique fonctionnel.
---

### Auteur

Projet réalisé dans le cadre d'une recherche R&D sur l'IA Quantique Hybride.