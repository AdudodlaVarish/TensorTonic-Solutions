def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:

    topkrelevant = len(set(recommended[:k]) & set(relevant))
    precision = topkrelevant / k 
    recall = topkrelevant / len(set(relevant)) 

    return [precision, recall]
    