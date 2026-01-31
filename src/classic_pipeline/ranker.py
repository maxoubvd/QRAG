import numpy as np

class ClassicalRanker:
    """
    Implémentation de MMR (Maximal Marginal Relevance) pour le ranking classique.
    Sert de point de comparaison (baseline) face à l'approche quantique QAOA.
    """
    def __init__(self, lambda_div=0.5):
        # Facteur d'équilibre Lambda :
        # 1.0 = Pertinence pure (risque de doublons d'information)
        # 0.0 = Diversité pure (risque de hors-sujet)
        # 0.5 = Compromis idéal
        self.lambda_div = lambda_div

    def rank(self, candidates, top_n=3):
        """
        Réordonne les candidats pour maximiser la diversité via l'algorithme MMR.
        """
        if not candidates:
            return []
            
        # 1. Initialisation : On sélectionne d'abord le document le plus pertinent (Score Cosine Max)
        # C'est notre point d'ancrage.
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # 2. Sélection itérative (Approche Gloutonne / Greedy)
        while len(selected) < top_n and remaining:
            best_score = -float('inf')
            next_selected = None
            
            for cand in remaining:
                # Pertinence intrinsèque (Score de similarité Query <-> Document)
                relevance = cand['score']
                
                # Redondance : On mesure la similarité maximale avec les documents DÉJÀ choisis.
                # Si le document ressemble trop à un doc existant, ce score sera élevé.
                redundancy = max([np.dot(cand['vector'], s['vector']) for s in selected])
                
                # Formule MMR : Score = (Lambda * Pertinence) - ((1 - Lambda) * Redondance)
                # On pénalise mathématiquement la redondance.
                mmr_score = (self.lambda_div * relevance) - ((1 - self.lambda_div) * redundancy)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    next_selected = cand
            
            if next_selected:
                selected.append(next_selected)
                remaining.remove(next_selected)
            else:
                break
                
        return selected