Voici les étapes pour tester le code sur un ordinateur quantique d'ibm

# 1. Se connecter/créer son compte IBM sur *https://quantum.cloud.ibm.com/*

Il faut avoir un ``IBMid``.

# 2. Une fois sur le dashboard **IBM Quantum Platform**

- Appuyer sur créer une instance
  - Choisir le plan gratuit et paramétrer le nom de l'instance
- Revenir sur le dashboard
- Appuyer sur créer une clé API 
  - Lui donner un nom
  - Cliquer sur créer
  - Copier et/ou télécharger la clé API

# 3. Ajouter la clé API dans le fichier ``.env``

``IBMQ_API_TOKEN=...``

# 4. Lancer la pipeline hardware

Dans le terminal taper:  
``python src/main_hardware.py``


# 5. Penser à récupérer les résultats

Vu qu'on peut faire tourner le code qu'une fois par mois faut vraiment penser à récupérer les résultats (genre les copier/coller quelque part).