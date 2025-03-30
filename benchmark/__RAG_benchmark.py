import numpy as np

class VectorSearchMetrics:
    def __init__(self, results, k=3):
        """
        :param results: dict, структура данных с результатами поиска
        :param k: int, количество топ-N документов для оценки
        """
        self.results = results
        self.k = k

    def recall_at_k(self):
        recalls = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                relevant = data["validation"]
                if relevant:
                    recalls.append(1)
                else:
                    recalls.append(0)
        return np.mean(recalls)

    def precision_at_k(self):
        precisions = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                search_ids = data["search_ids"][0][:self.k]
                relevant = data["validation"]
                if relevant:
                    precisions.append(1 / self.k)
                else:
                    precisions.append(0)
        return np.mean(precisions)

    def mrr(self):
        mrr_scores = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                position = data["position"]
                if position and position <= self.k:
                    mrr_scores.append(1 / position)
                else:
                    mrr_scores.append(0)
        return np.mean(mrr_scores)

    def hit_rate_at_k(self):
        hits = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                hits.append(1 if data["validation"] else 0)
        return np.mean(hits)

    def ndcg_at_k(self):
        def dcg(scores):
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))
        
        ndcg_scores = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                search_ids = data["search_ids"][0][:self.k]
                relevance = [1 if data["validation"] else 0 for _ in search_ids]
                dcg_score = dcg(relevance)
                idcg_score = dcg(sorted(relevance, reverse=True))
                ndcg_scores.append(dcg_score / idcg_score if idcg_score > 0 else 0)
        return np.mean(ndcg_scores)

    def run(self):
        """Запуск всех метрик """
        return {
            "Recall@K": self.recall_at_k(),
            "Precision@K": self.precision_at_k(),
            "MRR": self.mrr(),
            "Hit Rate@K": self.hit_rate_at_k(),
            "NDCG@K": self.ndcg_at_k()
        }