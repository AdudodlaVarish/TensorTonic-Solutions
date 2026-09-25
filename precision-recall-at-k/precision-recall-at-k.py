def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:

    recommended_ = set(recommended[:k])
    relevant_ = set(relevant)

    precision = len(recommended_ & relevant_) / len(recommended_)
    recall = len(recommended_ & relevant_) / len(relevant_)

    return [precision, recall]