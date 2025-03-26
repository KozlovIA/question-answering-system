import chromadb
from typing import List, Dict, Any, Callable
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions  
# import os

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ChromaDBManager:
    def __init__(self, storage_path: str, collection_name: str, embedding_function=None):
        """
        Инициализация клиента ChromaDB и загрузка коллекции.
        :param storage_path: Путь к директории для хранения данных.
        :param collection_name: Имя коллекции.
        :param embedding_function: Функция эмбеддингов (по умолчанию OpenAI, если не передана).
        """
        self.client = chromadb.PersistentClient(path=storage_path)
        
        # Если embedding_function не передан, используем встроенную функцию эмбеддингов ChromaDB
        if embedding_function is None:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.embedding_function = embedding_function  # Сохраняем функцию эмбеддингов как атрибут
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    def query(self, query_text: str, n_results: int = 2) -> Dict[str, Any]:
        """
        Выполняет запрос к коллекции.
        :param query_text: Текст запроса.
        :param n_results: Количество возвращаемых результатов.
        :return: Словарь с результатами поиска.
        """
        if self.embedding_function:
            query_vector = self.embedding_function([query_text])[0]
            return self.collection.query(query_embeddings=[query_vector], n_results=n_results)
        return self.collection.query(query_texts=[query_text], n_results=n_results)
    
    def insert_document(self, document_id: str, document_text: str, metadata: Dict[str, Any] = None):
        """
        Добавляет новый документ в коллекцию.
        :param document_id: Уникальный идентификатор документа.
        :param document_text: Текст документа.
        :param metadata: Метаданные (опционально).
        """
        data = {
            "ids": [document_id],
            "documents": [document_text],
            "metadatas": [metadata] if metadata else [{}]
        }
        if self.embedding_function:
            data["embeddings"] = [self.embedding_function([document_text])[0]]

        self.collection.add(**data)

    def update_document(self, document_id: str, new_text: str, new_metadata: Dict[str, Any] = None):
        """
        Полностью обновляет документ в коллекции по его идентификатору.
        :param document_id: Уникальный идентификатор документа.
        :param new_text: Новый текст документа.
        :param new_metadata: Новые метаданные (опционально).
        """
        # Удаляем старую версию документа
        self.collection.delete(ids=[document_id])
        
        # Добавляем обновлённый документ
        self.insert_document(document_id, new_text, new_metadata)

    def add_unique_documents(self, document_id: str, document_text: str, metadata: Dict[str, Any] = None):
        """
        Добавляет только новые документы в коллекцию ChromaDB.
        :param documents: Список документов.
        :param ids: Список id документов.
        """
        existing_ids = set(self.collection.get(ids=document_id, include=[])['ids'])
        
        new_docs = [(doc, doc_id) for doc, doc_id in zip(document_text, document_id) if doc_id not in existing_ids]
        
        if new_docs:
            new_documents, new_ids = zip(*new_docs)
            self.insert_document(document_id, document_text, metadata)
            return f"Добавлено {len(new_ids)} новых документов"
        else:
            return "Нет новых документов для добавления"

    
    def get_collection_keys(self) -> List[str]:
        """
        Возвращает ключи данных, содержащихся в коллекции.
        :return: Список ключей.
        """
        sample_query = self.query("test", 1)
        return list(sample_query.keys()) if sample_query else []
    

class CustomEmbeddingFunction:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализирует модель эмбеддингов.
        :param model_name: Название модели эмбеддингов (по умолчанию all-MiniLM-L6-v2).
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Создаёт эмбеддинги для списка текстов.
        :param input: Список строк (текстов).
        :return: Список эмбеддингов, где каждый эмбеддинг — список чисел (float).
        """
        return self.model.encode(input, convert_to_numpy=True)#.tolist()


if __name__ == "__main__":

    import yaml
    with open("config/chroma.yaml", "r") as file:
        config = yaml.safe_load(file)

    CHROMA_PATH = config["chroma_path"]
    COLLECTION_NAME = config["collection_name"]
    MODEL_NAME = config["model_name"]
    # Создаём менеджер с кастомной функцией эмбеддингов
    embedding_function = CustomEmbeddingFunction(model_name=MODEL_NAME)

    # Инициализируем менеджер с кастомными эмбеддингами
    db_manager = ChromaDBManager(
        storage_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    import json
    # Пример запроса
    query_results = db_manager.query("random forest", n_results=2)
    print("Search Results:", json.dumps(query_results, indent=4), sep='\n')
