# %%
from source.chroma_manager import ChromaDBManager
from source.llm_manager import LMStudioClient
from prompt.prompts import QuestionsToDoc
from tqdm import tqdm
import ast
import json
import logging

DB_CONFIG = LMStudioClient.load_config("config/embedding/questions_gen.yaml")
MODEL_CONFIG_PATH = "config/model_question_gen.yaml"

logging.basicConfig(filename="log/question_gen.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Инициализация менеджеров
source_chroma = ChromaDBManager(
    storage_path=DB_CONFIG["chroma_path"],
    collection_name=DB_CONFIG["collection_name"]
)

llm = LMStudioClient(MODEL_CONFIG_PATH)  # URL настроить под свой сервер

# Извлечение всех документов
all_documents = source_chroma.collection.get()
doc_ids = all_documents.get("ids", [])
doc_texts = all_documents.get("documents", [])
doc_metadata = all_documents.get("metadatas", [])

# %%
def safe_str_to_dict(s: str):
    """
    Преобразует строку в словарь, если возможно.
    В противном случае возвращает -1.
    """
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
        return False
    except (ValueError, SyntaxError):
        return False

# %%
for doc_id, doc_text, metadata in tqdm(zip(doc_ids, doc_texts, doc_metadata)):
    # Генерация вопросов через LLM
    prompt = QuestionsToDoc.QUESTION_FORMATION.format(document=doc_text)
    # print(repr(prompt))  # Выведет строку в raw-формате
    questions = llm.post_completion(user_input=prompt)
    llm.clear_context() # очищаем контекст
    questions_dict = safe_str_to_dict(questions)
    if questions_dict == False:
        prompt = QuestionsToDoc.CORRECTING_DICKTIONARY.format(input=questions)
        questions = llm.post_completion(user_input=prompt)
    
        questions_dict = safe_str_to_dict(questions)
        if questions_dict == False:
            logging.error(f" doc_id={doc_id}: questions_dict={questions_dict}")
            continue
        
    # Обновление метаданных
    metadata["questions"] = json.dumps(questions_dict)
    source_chroma.update_document(document_id=doc_id, new_text=doc_text, new_metadata=metadata)

print("Обновление коллекции завершено!")
