import yaml
import sys
import os
import re
import logging
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

# === Настройка логирования ===
logging.basicConfig(
    filename="log/error_arxiv.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if len(sys.argv) > 1:
    config_path = sys.argv[1]
    config_path = "config/embedding/" + config_path.split('.yaml')[0] + '.yaml'
else:
    config_path = "config/embedding/all-MiniLM-L6-v2.yaml"

with open(f"{config_path}", "r") as file:
    config = yaml.safe_load(file)

CHROMA_PATH = config["chroma_path"]
COLLECTION_NAME = config["collection_name"]
MODEL_NAME = config["model_name"]

DOCUMENTS_FOLDER = 'dataset_short'
TOPIC_PATH = 'config/topics_short.yaml'

# === Загрузка модели эмбеддингов ===
model = SentenceTransformer(MODEL_NAME)

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

def clean_text(text):
    text = replace_ligatures(text)  # Удаляем лигатуры
    text = re.sub(r'\f', '', text)  # Удаляем символы \f
    text = re.sub(r'(?m)^.$', '', text)  # Удаляем строки с одним символом
    text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)  # Убираем лишние переносы строк
    text = re.sub(r'(?<=\w)-\n', '', text)  # Убираем переносы слов
    text = re.sub(r'\n{2,}', '\n', text)  # Сводим подряд идущие переносы строк к одному
    text = re.sub(r'\bREFERENCES\b.*', '', text, flags=re.IGNORECASE | re.DOTALL)  # Удаляем все после REFERENCES
    text = re.sub(r'\d{4,}.*', '', text)  # Удаляем непонятные числовые строки
    text = re.sub(r'(?m)^\s*\d+\.?\s*$', '', text)  # Удаляем строки с номерами
    text = re.sub(r'(?m)^([A-Za-z]+\s*){1,3}\d+$', '', text)  # Удаляем табличные данные
    return text.strip()


def extract_text_from_pdf(file_path: str):
    """Извлекает текст из PDF файла с очисткой."""
    text = extract_text(file_path)
    cleaned_text = clean_text(text)
    return cleaned_text

def embed_texts(texts):
    """Создает эмбеддинги для списка текстов."""
    return model.encode(texts).tolist()

def upload_to_chromadb(documents, collection_name, db_path="./chroma_storage"):
    """Загружает документы в ChromaDB."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    for doc in documents:
        data = {
            "ids": [doc["ids"]],
            "documents": [doc["documents"]],
            "metadatas": [doc["metadata"]],
            "embeddings": [embed_texts([doc["documents"]])[0]]
        }
        collection.add(**data)
    
    print(f"Uploaded {len(documents)} documents to collection '{collection_name}'.")

def process_topic(topic):
    """Обрабатывает все документы по данной теме и загружает в ChromaDB."""
    topic_name = topic['name']
    folder = rf"{DOCUMENTS_FOLDER}\{topic_name}"
    all_documents = []
    
    for keyword in os.listdir(folder):
        folder_keywords = os.path.join(folder, keyword)
        print(folder_keywords)
        
        for file_name in os.listdir(folder_keywords):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(folder_keywords, file_name)
                try:
                    document_text = extract_text_from_pdf(file_path)
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
    upload_to_chromadb(all_documents, collection_name=COLLECTION_NAME, db_path=CHROMA_PATH)

def main():
    """Основная функция обработки тем."""
    topics = load_from_yaml(TOPIC_PATH)['topics']
    print(f"Loaded topics: {[topic['name'] for topic in topics]}")
    
    for topic in tqdm(topics):
        process_topic(topic)

if __name__ == "__main__":
    main()
