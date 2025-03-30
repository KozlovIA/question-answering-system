import numpy as np

class VectorSearchMetrics:
    def __init__(self, results):
        """
        :param results: dict, структура данных с результатами поиска
        """
        self.results = results

    def precision(self):
        """Метрика Precision — доля правильных документов среди всех найденных."""
        correct = 0
        total_found = 0  # Общее количество найденных документов
    
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                total_found += 1
                if data["validation"]:
                    correct += 1

        return correct / total_found if total_found > 0 else 0


    def mrr(self):
        """Метрика MRR — средний обратный ранг первого правильного документа."""
        mrr_scores = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                position = data["position"]
                if position is not False and position is not None:
                    mrr_scores.append(1 / (position + 1))  # Считаем индекс с 0 как первую позицию
                else:
                    mrr_scores.append(0)
        return np.mean(mrr_scores)

    
    def adjusted_score(self):
        """Метрика: 0.5 + 0.5 / (place + 1), где place — индекс позиции (начинается с 0)."""
        scores = []
        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                position = data["position"]
                if position is not False and position is not None:
                    scores.append(0.5 + 0.5 / (position + 1))  # Считаем индекс с 0 как первую позицию
                else:
                    scores.append(0)
        return np.mean(scores)
    
    def first_position_accuracy(self):
        """
        Вычисляет процент запросов, в которых релевантный документ был найден на 0-й позиции.
        
        :param results: dict, структура данных с результатами поиска
        :return: float, процент запросов с релевантным документом на 0-й позиции
        """
        total_queries = 0
        first_position_count = 0

        for doc_id, queries in self.results.items():
            for query, data in queries.items():
                total_queries += 1
                if str(data["position"]) == '0':
                    first_position_count += 1

        return (first_position_count / total_queries) * 100 if total_queries > 0 else 0


    def run(self, round_num=2):
        """Запуск всех метрик."""
        return {
            "Precision": round(self.precision(), round_num),
            "MRR": round(self.mrr(), round_num),
            "Adjusted Score": round(self.adjusted_score(), round_num),
            "First position accuracy": round(self.first_position_accuracy(), round_num)
        }
