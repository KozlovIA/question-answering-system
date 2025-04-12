# %% [markdown]
# ## IMPORT

# %%
import yaml
import sys
import os
import re
import logging
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from llama_cpp import Llama
import wordninja

# %% [markdown]
# ## Config


# %%
CONFIG_PATH = [#"config/embedding/questions_gen.yaml",
            "config/embedding/all-MiniLM-L6-v2.yaml",
            "config/embedding/all-mpnet-base-v2.yaml",
            "config/embedding/paraphrase-multilingual-MiniLM-L12-v2.yaml",
            "config/embedding/multi-qa-mpnet-base-dot-v1.yaml",
            "config/embedding/LaBSE.yaml",
            "config/embedding/distiluse-base-multilingual-cased-v1.yaml",
            "config/embedding/msmarco-distilbert-base-v4.yaml",
            "config/embedding/multi-qa-MiniLM-L6-cos-v1.yaml",
            "config/embedding/paraphrase-multilingual-mpnet-base-v2.yaml",
            "config/embedding/stsb-xlm-r-multilingual.yaml",
            "config/embedding/gtr-t5-large.yaml",
            "config/embedding/e5-large-v2.yaml",
            "config/embedding/multilingual-e5-large.yaml"]

# %%
# === Настройка логирования ===
logging.basicConfig(
    filename="log/error_arxiv.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_PATH = r"C:\Users\Igorexy\.lmstudio\models\MaziyarPanahi\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct.Q4_K_S.gguf"
TOKEN_TRESHOLD = 32768

DOCUMENTS_FOLDER = 'dataset'
TOPIC_PATH = 'config/topic.yaml'

def load_from_yaml(file_path: str):
    """Считывает YAML файл и возвращает список тем."""
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data



def replace_ligatures(text):
    ligatures = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
    "Æ": "AE", "Œ": "OE", "Ǆ": "DZ", "ǅ": "Dz",
    "Ϝ": "W", "Ϟ": "KS",
    "Ꜳ": "AA", "ꜳ": "aa", "Ꜵ": "AO"
}

    pattern = re.compile("|".join(re.escape(k) for k in ligatures))
    return pattern.sub(lambda m: ligatures[m.group()], text)


def remove_after_last_references(text):
    matches = list(re.finditer(r'\bREFERENCES\b', text, re.IGNORECASE))
    
    if matches:
        last_match = matches[-1]  # Берём последнее вхождение REFERENCES
        before_text = text[:last_match.start()]  # Текст до последнего REFERENCES
        after_text = text[last_match.end():]  # Текст после последнего REFERENCES
        
        # Условие: удаляем текст после, если его меньше, чем до
        if len(after_text) < len(before_text):
            return before_text  # Возвращаем только текст до последнего REFERENCES
        else:
            return text  # Если после REFERENCES текста больше или равно, ничего не удаляем
    
    return text  # Если REFERENCES нет, возвращаем исходный текст


def clean_text(text):
    text = replace_ligatures(text)  # Удаляем лигатуры
    text = re.sub(r'\f', '', text)  # Удаляем символы \f
    text = remove_after_last_references(text)   # Удаление ссылок на литературу
    text = re.sub(r'(?m)^.$', '', text)  # Удаляем строки с одним символом
    text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)  # Убираем лишние переносы строк
    text = re.sub(r'(?<=\w)-\n', '', text)  # Убираем переносы слов
    text = re.sub(r'\n{2,}', '\n', text)  # Сводим подряд идущие переносы строк к одному
    text = re.sub(r'\d{4,}.*', '', text)  # Удаляем непонятные числовые строки
    text = re.sub(r'(?m)^\s*\d+\.?\s*$', '', text)  # Удаляем строки с номерами
    text = re.sub(r'(?m)^([A-Za-z]+\s*){1,3}\d+$', '', text)  # Удаляем табличные данные
    return text.strip()


def extract_text_from_pdf(file_path: str):
    """Извлекает текст из PDF файла с очисткой."""
    text = extract_text(file_path)
    restored_text = " ".join(wordninja.split(text))     # восстановление пробелов
    cleaned_text = clean_text(text)
    return cleaned_text

llm_token_check = Llama(model_path=MODEL_PATH, n_ctx=32768, verbose=False)
def count_tokens_llama(text):
    return len(llm_token_check.tokenize(text.encode("utf-8"), add_bos=False))


def embed_texts(model, texts):
    """Создает эмбеддинги для списка текстов."""
    return model.encode(texts).tolist()

def upload_to_chromadb(documents, collection_name, model, db_path="./chroma_storage"):
    """Загружает документы в ChromaDB."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    for doc in documents:
        data = {
            "ids": [doc["ids"]],
            "documents": [doc["documents"]],
            "metadatas": [doc["metadata"]],
            "embeddings": [embed_texts(model, [doc["documents"]])[0]]
        }
        collection.add(**data)
    
    print(f"Uploaded {len(documents)} documents to collection '{collection_name}'.")




# %%
topics = load_from_yaml(TOPIC_PATH)['topics']
print(f"Loaded topics: {[topic['name'] for topic in topics]}")

DOCUMENTS_FOLDER = 'dataset'

all_documents = []
files_to_delete = []
check_token = False
token_treshold = 8000#TOKEN_TRESHOLD

for topic in tqdm(topics):

    topic_name = topic['name']
    folder = rf"{DOCUMENTS_FOLDER}\{topic_name}"
    if os.path.isdir(folder) == False:
        print(f"Папка {folder} не существует")
        continue

    for keyword in os.listdir(folder):
        folder_keywords = os.path.join(folder, keyword)
        print(folder_keywords)
        
        for file_name in os.listdir(folder_keywords):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(folder_keywords, file_name)
                try:
                    document_text = extract_text_from_pdf(file_path)

                    if check_token:     # Не записываем док, если превышает пороговое значение
                        if count_tokens_llama(document_text) > token_treshold:
                            files_to_delete.append(folder_keywords + "\\" + file_name)
                            continue   

                except Exception as e:
                    logging.error(f"Ошибка при считывании текста для темы '{topic_name}', ключевого слова '{keyword}': {e}")
                    print(f"Ошибка: {e} - при обработке темы '{topic_name}', ключевого слова '{keyword}'")
                    continue
                
                # Формируем записи для документа
                all_documents.append({
                    "ids": file_name.split('.pdf')[0],
                    "documents": document_text,
                    "metadata": {
                        "topic": topic_name,
                        "keyword": keyword,  # Добавляем ключевое слово в метаданные
                        "filename": file_name,
                    }
                })
    print(f"Extracted text from {len(all_documents)} documents for topic '{topic_name}'")

# print(files_to_delete)
# Удаление файлов
# for file_path in files_to_delete:
#     try:
#         os.remove(file_path)
#         print(f"Удалён: {file_path}")
#     except Exception as e:
#         print(f"Ошибка при удалении {file_path}: {e}")



# %%
# Удаление файлов
# for file_path in files_to_delete:
#     try:
#         os.remove(file_path)
#         print(f"Удалён: {file_path}")
#     except Exception as e:
#         print(f"Ошибка при удалении {file_path}: {e}")

# %%
# len(files_to_delete)

# # %%
# len(all_documents)

# %% [markdown]
# ### upload в БД

# %% [markdown]
# Загрузка

# %%

for conf_path in CONFIG_PATH:
    with open(f"{conf_path}", "r") as file:
        config = yaml.safe_load(file)

    CHROMA_PATH = config["chroma_path"]
    COLLECTION_NAME = config["collection_name"]
    MODEL_NAME = config.get("model_name", None)
    # MODEL_NAME = "all-MiniLM-L6-v2"

    # MODEL_PATH = config.get("model_path", r"C:\Users\Igorexy\.lmstudio\models\MaziyarPanahi\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct.Q4_K_S.gguf")
    # TOKEN_TRESHOLD = config.get("token_treshold", 32768)

    # DOCUMENTS_FOLDER = 'dataset'
    # TOPIC_PATH = 'config/topic.yaml'

    # %%
    # === Загрузка модели эмбеддингов ===
    model = SentenceTransformer(MODEL_NAME)


    upload_to_chromadb(all_documents, collection_name=COLLECTION_NAME, model=model, db_path=CHROMA_PATH)
