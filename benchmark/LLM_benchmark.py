import numpy as np
import re
import string
from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from bert_score import score as bert_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import nltk
from collections import Counter
from nltk.util import ngrams

nltk.download("punkt")
nltk.download('punkt_tab')


class AnswerEvaluator:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.rouge = Rouge()
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    def normalize(self, text):
        return re.sub(rf"[{string.punctuation}]", "", text.lower().strip())

    def exact_match(self, ref, pred):
        return int(self.normalize(ref) == self.normalize(pred))

    def f1_score(self, ref, pred):
        ref_tokens = self.normalize(ref).split()
        pred_tokens = self.normalize(pred).split()
        common = set(ref_tokens) & set(pred_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def rouge_l(self, ref, pred):
        return self.rouge.get_scores(pred, ref)[0]['rouge-l']['f']

    def bleu(self, ref, pred):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        smoothie = SmoothingFunction().method4
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

    def cosine_sim(self, ref, pred):
        embeddings = self.embedder.encode([ref, pred])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    def compute_rouge_n(reference, prediction, n=1):
        # Токенизация
        ref_tokens = nltk.word_tokenize(reference.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())

        # Генерация n-грамм
        ref_ngrams = list(ngrams(ref_tokens, n))
        pred_ngrams = list(ngrams(pred_tokens, n))

        # Подсчёт n-грамм
        ref_counts = Counter(ref_ngrams)
        pred_counts = Counter(pred_ngrams)

        # Подсчёт совпадений
        overlap = sum(min(ref_counts[gram], pred_counts[gram]) for gram in ref_counts)

        # Recall-ориентированный ROUGE-N
        if len(ref_ngrams) == 0:
            return 0.0
        return overlap / len(ref_ngrams)
        
    @staticmethod
    def aggregate_metrics(results):
        keys = ["f1", "rouge_l", "bleu", "cosine_similarity", "bertscore_f1", "quality_score"]
        metrics_avg = {}
        for key in keys:
            values = [res[key] for res in results]
            metrics_avg[key + "_avg"] = np.mean(values)
        return metrics_avg
    
    @staticmethod
    def compute_composite_score(res, weights=None):
        if weights is None:
            weights = {
                "bertscore_f1": 0.3,
                "cosine_similarity": 0.25,
                "rouge_l": 0.2,
                "f1": 0.15,
                "bleu": 0.05,
                # "exact_match": 0.05,
            }
        return sum(res[k] * weights[k] for k in weights)


    def evaluate(self):
        results = []
        generation_times = []

        refs, preds = [], []
        for entry in self.data:
            qa = entry["question_1"]
            ref = qa["answer"]
            pred = qa["answer_llm"]

            refs.append(ref)
            preds.append(pred)

            generation_times.append(qa["generation_time"])

            results.append({
                "id": entry["id"],
                # "exact_match": self.exact_match(ref, pred),
                "f1": self.f1_score(ref, pred),
                "rouge_l": self.rouge_l(ref, pred),
                "bleu": self.bleu(ref, pred),
                "cosine_similarity": self.cosine_sim(ref, pred),
            })

        # BERTScore (batched for efficiency)
        P, R, F1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
        for i in range(len(results)):
            results[i]["bertscore_f1"] = F1[i].item()

        # Time-based metrics
        gen_times = np.array(generation_times)
        speed_metrics = {
            "gen_time_avg": gen_times.mean(),
            "gen_time_median": np.median(gen_times),
            "gen_time_95th_percentile": np.percentile(gen_times, 95),
        }

        for res in results:
            res["quality_score"] = self.compute_composite_score(res, weights=None)

        agg_metrics = self.aggregate_metrics(results)

        return agg_metrics, results, speed_metrics

