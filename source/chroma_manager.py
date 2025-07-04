import chromadb
from typing import List, Dict, Any, Callable
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions  
import os
import yaml

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ChromaDBManager:
    def __init__(self, config_path: str):
        """
        Инициализация клиента ChromaDB и загрузка коллекции на основе конфигурационного файла.
        :param config_path: Путь к YAML-файлу конфигурации.
        """
        config = self.load_config(config_path)
        storage_path = config.get("chroma_path", "./chroma_storage")
        collection_name = config.get("collection_name", "default_collection")
        model_name = config.get("model_name", "all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(path=storage_path)
        
        # embedding_function = embedding_functions.DefaultEmbeddingFunction(model_name=model_name)
        embedding_function = CustomEmbeddingFunction(model_name=model_name)
        
        self.embedding_function = embedding_function  # Сохраняем функцию эмбеддингов как атрибут
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    @staticmethod
    def load_config(file_path: str):
        """Считывает YAML-файл конфигурации."""
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    
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

    def get_collection_keys(self) -> List[str]:
        """
        Возвращает ключи данных, содержащихся в коллекции.
        :return: Список ключей.
        """
        sample_query = self.query("test", 1)
        return list(sample_query.keys()) if sample_query else []
    

    def get_all_ids(self) -> List[str]:
        """
        Возвращает список всех идентификаторов в коллекции.
        :return: Список строк-идентификаторов.
        """
        results = self.collection.get()
        return results.get("ids", [])

    def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает все документы в виде словаря: {id: {"document": ..., "metadata": ...}, ...}
        """
        results = self.collection.get(include=["documents", "metadatas"], limit=None)
        return {
            doc_id: {
                "document": doc,
                "metadata": meta
            }
            for doc_id, doc, meta in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", [])
            )
        }

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает документы по списку идентификаторов в виде словаря: {id: {"document": ..., "metadata": ...}, ...}
        """
        results = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            doc_id: {
                "document": doc,
                "metadata": meta
            }
            for doc_id, doc, meta in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", [])
            )
        }
    
    def get_full_collection(self, include_embeddings: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает всю коллекцию: документы, метаданные и (опционально) эмбеддинги.
        """
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        results = self.collection.get(include=include, limit=None)

        output = {}
        if include_embeddings:
            for doc_id, doc, meta, emb in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("embeddings", [])
            ):
                output[doc_id] = {
                    "document": doc,
                    "metadata": meta,
                    "embedding": emb
                }
        else:
            for doc_id, doc, meta in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", [])
            ):
                output[doc_id] = {
                    "document": doc,
                    "metadata": meta
                }
        return output

    

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

    # with open("config/chroma.yaml", "r") as file:
    #     config = yaml.safe_load(file)

    # CHROMA_PATH = config["chroma_path"]
    # COLLECTION_NAME = config["collection_name"]
    # MODEL_NAME = config["model_name"]
    # Создаём менеджер с кастомной функцией эмбеддингов
    # embedding_function = CustomEmbeddingFunction(model_name=MODEL_NAME)

    # Инициализируем менеджер с кастомными эмбеддингами
    db_manager = ChromaDBManager(
        config_path=r"E:\ImportantFiles\Documents\University\Magic App\config\embedding\e5-large-v2.yaml"
    )
    
    import json
    # Пример запроса
    query_results = db_manager.query("random forest", n_results=2)
    print("Search Results:", json.dumps(query_results, indent=4), sep='\n')
