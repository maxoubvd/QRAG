import os
# Protection des variables d'environnement pour éviter les conflits entre TensorFlow et PyTorch
os.environ["USE_TORCH"] = "TRUE"
os.environ["USE_TF"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import pickle
import pennylane as qml

print("\n" + "="*40)
print("🩺 DIAGNOSTIC QRAG - TEST UNITAIRE")
print("="*40)

# TEST 1 : LE MOTEUR QUANTIQUE (HELLO WORLD)
# Objectif : Vérifier que PennyLane et l'amplification de Grover fonctionnent
# sur un cas minuscule (3 qubits, 8 états) indépendant de nos données.
print("\n[TEST 1] Vérification du Moteur PennyLane (Grover 3 qubits)...")
try:
    dev = qml.device("default.qubit", wires=3)
    @qml.qnode(dev)
    def tiny_grover():
        # Superposition uniforme (Hadamard sur tous les qubits)
        for i in range(3): qml.Hadamard(wires=i)
        
        # Oracle : Marquer l'état |111> (Index 7)
        # FlipSign est l'opérateur natif le plus robuste pour l'inversion de phase.
        qml.FlipSign(7, wires=[0, 1, 2])
        
        # Diffuseur Atomique (Opérateur de Grover)
        # Structure mathématique : H -> Réflexion autour de 0 (Flip0) -> H
        for i in range(3): qml.Hadamard(wires=i)
        qml.FlipSign(0, wires=[0, 1, 2])
        for i in range(3): qml.Hadamard(wires=i)
        
        return qml.probs(wires=[0, 1, 2])
    
    probs = tiny_grover()
    target_prob = probs[7]
    print(f"   > Probabilité de l'état cible |111> : {target_prob:.4f}")
    
    if target_prob > 0.5:
        print("   SUCCÈS : L'algorithme de Grover fonctionne sur cette machine.")
    else:
        print("   ÉCHEC : PennyLane n'amplifie pas. Problème d'installation ou de version.")
except Exception as e:
    print(f"   CRASH : {e}")


# TEST 2 : INTÉGRITÉ DES DONNÉES
# Objectif : S'assurer que le dataset vectorisé n'est pas corrompu (NaN, Zéros)
# avant de l'injecter dans le circuit.
print("\n[TEST 2] Vérification des Données (processed_data.pkl)...")
try:
    with open("data/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    vecs = data["vectors"]
    print(f"   > Dimensions des vecteurs : {vecs.shape}")
    
    # Vérification des NaNs ou Zéros
    if np.isnan(vecs).any():
        print("   ÉCHEC : Les vecteurs contiennent des NaN (Erreur Data Prep).")
    elif np.all(vecs == 0):
        print("   ÉCHEC : Les vecteurs sont tous à zéro.")
    else:
        print("   SUCCÈS : Les données semblent saines.")
        
    # Vérification PCA
    sample_norm = np.linalg.norm(vecs[0])
    print(f"   > Norme du premier vecteur : {sample_norm:.4f} (Doit être ~1.0)")

except Exception as e:
    print(f"   ERREUR FICHIER : {e}")


# TEST 3 : L'ORACLE EN CONDITIONS RÉELLES
# Objectif : Vérifier que notre classe QuantumGroverRetriever parvient bien
# à identifier et amplifier un document spécifique parmi les 2000+.
print("\n[TEST 3] Simulation de l'Oracle QRAG (Sans Diffuseur)...")
try:
    # Ajout du chemin src pour importer le module local
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from quantum_pipeline.simulation.grover_retriever import QuantumGroverRetriever
    
    retriever = QuantumGroverRetriever()
    # On force une requête bidon pour voir si l'Oracle réagit
    dummy_query = np.random.rand(retriever.n_docs) 
    # (Note: ici on triche, on passe directement un vecteur similaire aux docs pour tester)
    target_idx = 42
    dummy_vec = retriever.vectors[target_idx]
    
    # On espère que l'Oracle va marquer l'index 42
    print(f"   > Test de marquage sur l'index {target_idx}...")
    
    # 1. Test de l'Oracle seul (Phase check - invisible en probabilité)
    @qml.qnode(retriever.dev)
    def oracle_check():
        for i in range(retriever.n_qubits): qml.Hadamard(wires=i)
        
        # On appelle l'Oracle
        retriever.get_oracle(dummy_vec, num_to_mark=1)
        
        # Astuce : On mesure la valeur attendue de PauliZ sur tous les qubits
        # Si l'état est marqué (-1), cela devrait bouger légèrement l'espérance globale
        return qml.probs(wires=range(retriever.n_qubits))

    probs = oracle_check()
    # Si l'oracle marche, l'état 42 ne doit PAS avoir changé de probabilité (phase only)
    # MAIS si on applique le Diffuseur ensuite...
    
    # 2. Test Complet (Oracle + Diffuseur -> Amplification visible)
    @qml.qnode(retriever.dev)
    def full_check():
        for i in range(retriever.n_qubits): qml.Hadamard(wires=i)
        retriever.get_oracle(dummy_vec, num_to_mark=1)
        retriever.get_diffusion() # On teste le diffuseur atomique
        return qml.probs(wires=range(retriever.n_qubits))
        
    probs_full = full_check()
    prob_42 = probs_full[target_idx]
    
    print(f"   > Probabilité de l'index 42 après 1 itération : {prob_42:.6f}")
    # Calcul de la probabilité au hasard (1 chance sur N)
    baseline = 1/retriever.n_docs
    print(f"   > Baseline (Uniforme) : {baseline:.6f}")
    
    # Si la probabilité a au moins doublé, la mécanique fonctionne.
    if prob_42 > baseline * 2:
        print("   SUCCÈS : L'amplification a lieu !")
    else:
        print("   ÉCHEC : Le graphique plat est confirmé. L'Oracle ou le Diffuseur ne mord pas.")

except Exception as e:
    print(f"   ERREUR CODE : {e}")

print("\n" + "="*40)