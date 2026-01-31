### 1. L'Objectif : Trouver une aiguille dans une botte de vecteurs

Dans notre cas :

* **La botte de foin ($N$)** : Les 2136 documents (chunks).
* **L'aiguille ($w$)** : Les documents sémantiquement proches de la question.
* **L'outil** : L'algorithme de Grover qui, au lieu de vérifier chaque document un par un (Classique $O(n)$), fait "monter" la probabilité des bons documents en utilisant des interférences ($O(\sqrt n)$).

---

### 2. L'Architecture : Ce que fait ton code étape par étape

Regardons la méthode `search()` et ses composants dans `grover_retriever.py`.

#### A. L'Espace de Recherche (Les Qubits)

Le code calcule : `self.n_qubits = int(np.ceil(np.log2(self.n_docs)))`.

* Pour 2136 documents, nous allouons **12 qubits**.
* Pourquoi ? Parce que $2^{12} = 4096$ états possibles. Chaque état quantique de base $\ket{00...00}$ à $\ket{11...11}$ représente **l'index** d'un document.
* *Différence avec la théorie pure de l'algorithme de Grover:* Ici, nous n'encodons pas le *texte* du document dans les qubits, mais son *adresse* (son index de 0 à 2135).

#### B. L'Initialisation (Superposition)

```python
for i in range(self.n_qubits): qml.Hadamard(wires=i)
```

Au départ, le système est dans une superposition uniforme. Cela signifie que la probabilité de trouver n'importe quel document est égale ($\frac{1}{2136} \approx 0.0004$). C'est le "bruit de fond" gris sur le graphique.

#### C. L'Oracle "Hybride" (Le Cœur du système)

C'est ici que se trouve la plus grande astuce de l'implémentation.

**En théorie :** L'Oracle est une boîte noire quantique qui regarde l'état $\ket{x}$, calcule une fonction $f(x)$ (est-ce que ce document répond à la question ?) et inverse la phase si oui.
**Dans notre code (`get_oracle`) :** Construire un circuit quantique qui fait des produits scalaires de vecteurs 64D serait monstrueusement complexe (des milliers de portes).
**Notre solution: Oracle Hybride :**

1. **Calcul Classique :** On calcule la similarité (dot product) avec Numpy *avant* d'appliquer les portes quantiques.
```python
similarities = np.dot(self.vectors, query_vector)
indices_to_mark = ... # Les top-k meilleurs
```


2. **Marquage Quantique :** Une fois que tu sais que l'on connaît les indices des document #496 est une cible, tu demandes au circuit quantique de changer sa phase.
```python
qml.FlipSign(int(idx), wires=range(self.n_qubits))
```


*Ce que ça fait physiquement :* L'amplitude de l'état $\ket{496}$ passe de $+\alpha$ à $-\alpha$. C'est ce qu'on appelle le "Phase Kickback".

#### D. Le Diffuseur (L'Amplificateur)

Une fois la phase inversée, les probabilités n'ont pas encore changé (car $|-\alpha|^2=|\alpha|^2$). Il faut le **Diffuseur** pour convertir cette différence de phase en différence d'amplitude (probabilité).

Notre implémentation `get_diffusion` est la version "Atomique" très efficace :

1. **Hadamard (`H`)** : Change de base.
2. **Réflexion (`FlipSign(0)`)** : Inverse le signe de l'état $\ket{00...0}$. Mathématiquement, cela équivaut à la matrice de diffusion de Grover $2\ket{\psi}\bra{\psi}-I$.
3. **Hadamard (`H`)** : Revient à la base d'origine.

**Résultat :** L'amplitude moyenne baisse, et l'amplitude de ta cible (qui avait une phase négative) s'inverse à nouveau par rapport à la moyenne, devenant **très grande**.

---

### 3. Différences Clés : Théorie de l'algorithme de Grover vs Notre Code

Voici le tableau de vérité pour comprendre nos choix d'implémentation :

| Concept | Théorie Standard (Livres) | Ton Implémentation (QRAG) | Pourquoi ce choix ? |
| --- | --- | --- | --- |
| **Oracle** | Circuit quantique complet qui calcule $f(x)$ arithmétiquement. | **Oracle Hybride** : Calcul classique de la similarité, injection du résultat via `FlipSign`. | Simuler l'arithmétique flottante 64D en quantique est impossible sur un simulateur léger. C'est une approche standard pour les PoC NISQ. |
| **Itérations** | Exactement $\frac{\pi}{4}\sqrt{N}$. | **Adaptatif & Plafonné** : Calcul théorique mais limité à 5-10 itérations. | Pour éviter l'over-rotation et garder la démo rapide (< 1s), tout en prouvant l'amplification. |
| **Portes** | Décomposition en portes CNOT, Toffoli, X, Z. | **Portes de Haut Niveau** : Utilisation de `qml.FlipSign` (Multi-controlled Z abstrait). | Évite les erreurs de précision numérique, réduit la profondeur apparente, accélère `lightning.qubit`. |
| **Données** | QRAM (Mémoire Quantique) théorique. | **Indexation** : Les qubits représentent des indices, pas les données elles-mêmes. | Nous n'avons pas de QRAM physique. L'encodage par index est la seule méthode viable actuellement. |

---

### 4. Résumé Visuel du Fonctionnement

On peut imaginer une immense table de mixage avec 4096 curseurs (les documents).

1. **Initialisation :** Tous les curseurs sont réglés à la même hauteur (très bas).
2. **Oracle (nous) :** On regarde nos notes (calcul classique Numpy), on repère les curseurs 496 et 585, et on tire **vers le bas** leur mécanisme interne (Phase Inversion). *Visuellement, rien ne bouge encore.*
3. **Diffuseur (La Machine) :** On appuie sur le bouton "Grover". La machine calcule la moyenne de tous les curseurs. Elle inverse tout le monde autour de cette moyenne.
* Comme les curseurs 496 et 585 étaient "négatifs" (phase inversée), l'inversion autour de la moyenne les propulse **très haut vers le haut**.
* Les autres curseurs descendent légèrement pour compenser.


4. **Mesure :** On regarde la table. Les curseurs 496 et 585 crèvent le plafond. On a notre réponse.

C'est exactement ce que le graphique `grover_final_check.png` a montré : le pic bleu (curseur propulsé) au milieu du bruit gris.