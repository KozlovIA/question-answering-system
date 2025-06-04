# %%
from source.chroma_manager import ChromaDBManager, CustomEmbeddingFunction
from tqdm import tqdm
import yaml
import sys
import json 
import pandas as pd
import os


LANG = '_RUS'
# LANG = ''

# %% [markdown]
# ### config

# %%
CONFIG_PATH_LIST = [
    "config/embedding/all-MiniLM-L6-v2.yaml",
    "config/embedding/all-mpnet-base-v2.yaml",
    # "config/embedding/paraphrase-multilingual-MiniLM-L12-v2.yaml",         ### не юзали 
    # # "config/embedding/multi-qa-mpnet-base-dot-v1.yaml",       ### не юзали  
    "config/embedding/LaBSE.yaml",
    # # "config/embedding/distiluse-base-multilingual-cased-v1.yaml",     ### не юзали 
    # "config/embedding/msmarco-distilbert-base-v4.yaml",
    "config/embedding/multi-qa-MiniLM-L6-cos-v1.yaml",
    "config/embedding/paraphrase-multilingual-mpnet-base-v2.yaml",
    # "config/embedding/stsb-xlm-r-multilingual.yaml",      ### не юзали 
    # "config/embedding/gtr-t5-large.yaml",
    "config/embedding/e5-large-v2.yaml",
    "config/embedding/multilingual-e5-large.yaml"
] 

CONFIG_PATH_QUESTION = "config/embedding/questions_rus.yaml"


    # %%
for CONFIG_PATH in CONFIG_PATH_LIST:
    print(CONFIG_PATH)
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    CHROMA_PATH = config["chroma_path"]
    COLLECTION_NAME = config["collection_name"]
    MODEL_NAME = config["model_name"]


    # Создаём менеджер с кастомной функцией эмбеддингов
    embedding_function = CustomEmbeddingFunction(model_name=MODEL_NAME)

    # Инициализируем менеджер с кастомными эмбеддингами
    model_chroma = ChromaDBManager(CONFIG_PATH)

    # Инициализируем менеджер с вопросами
    questions_chroma = ChromaDBManager(CONFIG_PATH_QUESTION)

    # Извлечение всех документов
    all_documents = questions_chroma.collection.get()
    doc_ids = all_documents.get("ids", [])
    doc_metadata = all_documents.get("metadatas", [])
    len(doc_ids)

    # %%
    # doc_ids_batch = doc_ids
    # doc_metadata_batch = doc_metadata

    # %%
    result = {}

    # # Прогон идет по БД с вопросами
    for doc_id, metadata in tqdm(zip(doc_ids, doc_metadata)): 

        question_1 = eval(metadata['questions'])['question_1']
        # question_2 = eval(metadata['questions'])['question_2']
        answer_1 = eval(metadata['questions'])['answer_1']
        # answer_2 = eval(metadata['questions'])['answer_2']
        query_results_1 = model_chroma.query(question_1, n_results=3)
        # query_results_2 = model_chroma.query(question_2, n_results=3)
        
        position_1, position_2 = False, False

        validation_1 = doc_id in query_results_1['ids'][0]
        if validation_1:
            position_1 = query_results_1['ids'][0].index(doc_id)

        # validation_2 = doc_id in query_results_2['ids'][0]
        # if validation_2:
        #     position_2 = query_results_2['ids'][0].index(doc_id)
        
        temp_res = {
            doc_id: {
                "question_1": {
                        "search_ids": query_results_1['ids'],
                        "validation": validation_1,
                        "position": position_1,
                        "question": question_1,
                        "answer": answer_1,
                                },
                # "question_2": {
                #         "search_ids": query_results_2['ids'],
                #         "validation": validation_2,
                #         "position": position_2,
                #         "question": question_2,
                #         "answer": answer_2,
                # }
            }
        }
        result.update(temp_res)

    os.makedirs(f"benchmark/output_RAG{LANG}", exist_ok=True)

    output_filemane = f"benchmark/output_RAG{LANG}/{COLLECTION_NAME}.json"

    with open(output_filemane, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # %% [markdown]
    # # Расчет метрик

    # %%
    from benchmark.RAG_benchmark import VectorSearchMetrics

    filename = f"benchmark/output_RAG{LANG}/{COLLECTION_NAME}"
    with open(f"{filename}.json", encoding="utf-8") as file:
        data = json.load(file)  # Преобразование JSON в словарь

    bench = VectorSearchMetrics(data)
    metrics = bench.run()

    pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_excel(f"{filename}.xlsx", index=False)

# %% [markdown]
# # Составление единой таблицы сравнения

# %%
import os
import pandas as pd

# Путь к папке с Excel файлами
folder_path = fr'benchmark\output_RAG{LANG}'

# Список для хранения данных
data = []

# Перебираем все файлы в папке
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        # Полный путь к файлу
        file_path = os.path.join(folder_path, filename)
        
        # Читаем таблицу из файла
        df = pd.read_excel(file_path)
        
        # Получаем имя модели из имени файла (без расширения)
        model_name = os.path.splitext(filename)[0].replace("-embedding", '')
        
        # Добавляем столбец с названием модели
        df['Model'] = model_name
        
        # Добавляем данные в общий список
        data.append(df)

# Объединяем все данные в одну таблицу
result_df = pd.concat(data, ignore_index=True)

# Переводим данные в удобный формат для сравнения
pivot_df = result_df.pivot(index='metric', columns='Model', values='value')

# Сортируем
pivot_df = pivot_df.T.sort_values("Precision", ascending=False).T.reindex(["Precision", "Adjusted Score", "MRR", "First position accuracy"])

# Сохраняем итоговую таблицу в новый Excel файл
pivot_df.to_excel(f'benchmark/comparison_models{LANG}.xlsx')
