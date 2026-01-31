Si Grover était un "Chercheur" (qui fouille une botte de foin), QAOA est un **"Stratège"**. Son rôle n'est pas de trouver un document, mais de **composer la meilleure équipe** de documents possible.

Voici l'anatomie de notre `GroQRanker` dans le fichier `qaoa_ranker.py`.

---

### 1. L'Objectif : Le Dilemme du Bibliothécaire

On a récupéré (via Grover) disons 8 documents candidats.

* **Problème :** Si on donne les 8 au LLM, il peut y avoir des doublons (3 fois la même info) ou du bruit.
* **But :** En garder seulement 3 ou 4 qui sont :
1. **Pertinents** (proches de la question).
2. **Diversifiés** (qui ne disent pas la même chose).


* **La Méthode :** C'est un problème d'**Optimisation Combinatoire**. Parmi toutes les combinaisons possibles de documents, laquelle est la "meilleure" ?

---

### 2. La Traduction Physique : Le Modèle d'Ising

Pour qu'un ordinateur quantique résolve ce problème, il faut le transformer en un problème de **physique**. En physique, tout système tend vers son état d'énergie minimale (l'état fondamental).

Notre code transforme le "Score de Ranking" en "Énergie".

* **Haute Pertinence** = Énergie Basse (On aime ça).
* **Haute Redondance** = Énergie Haute (On évite ça, ça coûte cher).

Regardons `build_hamiltonian` :

#### A. Les Termes Linéaires (La Pertinence)

```python
coeffs.append(-relevance[i] / 2)
obs.append(qml.PauliZ(i))
```

* **PauliZ ($Z$)** est un opérateur qui regarde si un qubit est à 0 (non choisi) ou 1 (choisi).
* Le signe **négatif** (`-relevance`) est crucial : on veut **minimiser** l'énergie totale. Donc si un document très pertinent est choisi (état 1), il apporte une grande valeur négative, ce qui réduit l'énergie du système. C'est bon !

#### B. Les Termes Quadratiques (La Redondance)

```python
coeffs.append(self.lambda_div * redundancy_matrix[i][j] / 4)
obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
```

* Ici, on regarde l'interaction entre deux documents ($i$ et $j$).
* `PauliZ(i) @ PauliZ(j)` s'active si les deux sont dans le même état (tous les deux choisis).
* Le coefficient est **positif** (`+ redundancy`). Cela crée une "pénalité". Si tu choisis deux documents très similaires ensemble, l'énergie du système grimpe. Le système va donc "naturellement" éviter de les mettre ensemble pour rester à basse énergie.

C'est ce qu'on appelle un **Hamiltonien de Coût ($H_c$)**.

---

### 3. Le Circuit QAOA : Le "Mixeur" Quantique

Maintenant que le paysage énergétique est défini (les creux sont les bonnes solutions, les bosses sont les mauvaises), comment trouver le creux le plus profond ?

C'est le rôle de la méthode `circuit` et de la fonction `rank`.

#### A. Superposition (Le départ neutre)

```python
for i in range(num_nodes): qml.Hadamard(wires=i)
```

On commence par considérer **toutes les combinaisons possibles** de documents avec une probabilité égale. C'est l'état .

#### B. L'Alternance (La danse du QAOA)

L'algorithme alterne deux étapes, répétées `n_layers` fois (comme un four à plusieurs étapes de cuisson) :

1. **Phase Cost (`H_cost`, paramètre $\gamma$) :**
```python
qml.ApproxTimeEvolution(H_cost, gammas[i], 1)
```


Ici, on applique notre grille de lecture "Pertinence/Diversité". Les "bonnes" combinaisons voient leur phase quantique changer. On creuse les solutions.
2. **Phase Mixer (`H_mixer`, paramètre $\beta$) :**
```python
qml.ApproxTimeEvolution(H_mixer, betas[i], 1)
```


Si on ne faisait que l'étape 1, on resterait bloqué. Le Mixer (basé sur Pauli X) permet aux qubits de "sauter" d'un état à l'autre (flip de 0 à 1 et inversement). C'est l'exploration.

---

### 4. Différences Clés : Théorie vs Ton Code

Comme pour Grover, il y a une différence majeure entre la théorie académique et notre implémentation "Temps Réel".

| Concept | Théorie Standard (Vrai QAOA) | Notre Implémentation (GroQ-Rank) | Pourquoi ce choix ? |
| --- | --- | --- | --- |
| **Boucle** | **Hybride (Boucle Fermée)** : On lance le circuit, on mesure l'énergie, un optimiseur classique (ex: COBYLA) modifie $\gamma$ et $\beta$, et on recommence 100 fois. | **Open Loop (One-Shot)** : On lance le circuit une seule fois avec des paramètres $\gamma$ et $\beta$ fixés ou aléatoires. | Une boucle d'optimisation complète prendrait trop de temps (>30s) pour une démo interactive. |
| **Résultat** | On cherche l'état d'énergie minimale absolue. | On cherche une "bonne" solution amplifiée par l'interférence constructive. | Avec de bons paramètres par défaut, QAOA agit comme un filtre puissant même sans convergence parfaite. |
| **Qubits** | Qubits physiques. | `default.qubit` (Python). | Ici on a peu de qubits (5 à 8 documents candidats), donc `default.qubit` est assez rapide, pas besoin de `lightning`. |

---

### 5. Résumé Visuel du Fonctionnement

Imagine que n'on doive faire une playlist de 3 chansons pour une soirée.

1. **Hamiltonien de Coût (Les règles) :**
* Si la chanson est "dance" (Pertinence), on gagnes des points.
* Si on met deux chansons du même artiste (Redondance), on perds des points.


2. **Superposition :** On étale tous tes CD sur le sol. Toutes les combinaisons sont possibles.
3. **QAOA Layer 1 (Coût) :** On "marque" mentalement les playlists qui respectent les règles.
4. **QAOA Layer 1 (Mixer) :** On mélange un peu les cartes pour essayer des variantes (et si je mettais celle-là à la place ?).
5. **Répétition :** On refait ça 2 fois.
6. **Mesure :** À la fin, l'interférence quantique fait que la playlist la plus probable qui ressort est celle qui maximise l'ambiance (Pertinence) sans lasser les gens (Diversité).

Le résultat est un **Bitstring**, par exemple `11001000`.
Cela veut dire : "Garde le doc 1, le doc 2 et le doc 5. Jette les autres".

C'est ainsi que notre **GroQ-Ranker** nettoie les résultats de Grover avant de les envoyer au LLM.