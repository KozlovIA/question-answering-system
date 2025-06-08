import gradio as gr
import os
import shutil
import tempfile
import re
from source.chroma_manager import ChromaDBManager
from source.llm_manager import LMStudioClient
from prompt.prompts import QA_context

from gradio.themes.utils import colors
from gradio.themes.base import Base

from html import escape
import logging

# Настройка логгера
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename='log/gradio_QA.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

class LightTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            neutral_hue=colors.gray,
            font=["Arial", "sans-serif"]
        )
        self.set(
            body_background_fill="white",
            body_text_color="black",
            block_background_fill="white",
            block_border_color="lightgray",
            button_primary_background_fill="deepskyblue",
            button_primary_text_color="white"
        )


class ChatInterface():
    def __init__(self, chroma_config_path: str, llm_config_path: str, lang: str='en'):
        self.chat_history = []  # История сообщений
        self.chroma_db = ChromaDBManager(config_path=chroma_config_path)
        self.llm_client = LMStudioClient(config_path=llm_config_path)
        self.setup_interface()
        self.lang = lang

    def find_pdf_path_by_id(self, doc_id: str) -> str:
        base_dir = "dataset"
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.startswith(doc_id) and file.endswith(".pdf"):
                    return os.path.join(root, file)
        return ""
    
    @staticmethod
    def remove_think_block(text: str) -> str:
        """
        Удаляет блок <think>...</think> вместе с содержимым.
        Возвращает остальной текст.
        """
        # Удаляет первый блок <think>...</think> и всё внутри (включая перевод строк)
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
    
    def send_message(self, message, use_rag):
        try:
            context = "The context is empty, use your knowledge"
            system = f"{message}"
            retrieved_ids = []
            download_path = ""

            if use_rag:
                rag_results = self.chroma_db.query(query_text=message, n_results=2)
                retrieved_docs = rag_results.get("documents", [[]])[0]
                retrieved_ids = rag_results.get("ids", [[]])[0]
                
                # ✅ Логируем retrieved_ids
                logging.debug(f"[RAG] Retrieved IDs: {retrieved_ids}")

                context = "\n".join(retrieved_docs)
                system = QA_context.SYSTEM_QA_SHORT_RUS.format(question=message, context=context) if self.lang == 'ru' else QA_context.SYSTEM_QA_SHORT.format(question=message, context=context)

                temp_dir = tempfile.mkdtemp()
                for doc_id in retrieved_ids:
                    pdf_path = self.find_pdf_path_by_id(doc_id)
                    if pdf_path:
                        shutil.copy(pdf_path, os.path.join(temp_dir, os.path.basename(pdf_path)))

                archive_path = shutil.make_archive(temp_dir, 'zip', temp_dir)
                safe_path = os.path.join("downloads", os.path.basename(archive_path))
                os.makedirs("downloads", exist_ok=True)
                shutil.copy(archive_path, safe_path)

                download_path = safe_path

            full_prompt = system
            response = self.llm_client.post_completion(full_prompt)
            response = self.remove_think_block(response)
            
            # ✅ Логируем response от LLM
            logging.debug(f"[LLM] Response: {repr(response)}")

            # ✅ Логируем финальный ответ, отправляемый пользователю
            logging.debug(f"[ChatHistory] User: {message}\nBot: {response}")

            if not response or not isinstance(response, str):
                response = "⚠️ Получен пустой или некорректный ответ от модели."

            self.chat_history.append((message, response))

        except Exception as e:
            response = f"Произошла ошибка: {str(e)}"
            logging.error(f"[ERROR] {response}")
            self.chat_history.append((message, response))
            retrieved_ids = []
            download_path = ""

        return self.chat_history, "", "\n".join(retrieved_ids), download_path if download_path and os.path.exists(download_path) else None


    
    def clear_chat(self):
        """Очищает историю чата."""
        self.chat_history = []
        self.llm_client.clear_context()
        return self.chat_history
    
    def setup_interface(self):
        """Настраивает интерфейс Gradio."""
        with gr.Blocks(theme=LightTheme()) as self.app:
            self.chatbot = gr.Chatbot(height=800, render_markdown=True)
            self.input_text = gr.Textbox(label="Введите сообщение")
            self.use_rag_checkbox = gr.Checkbox(label="Использовать RAG", value=False)

            self.retrieved_ids_box = gr.Textbox(label="IDs извлечённых документов", interactive=False, lines=3)
            self.download_button = gr.File(label="Скачать документы", interactive=False)

            with gr.Row():
                self.send_button = gr.Button("Отправить")
                self.clear_button = gr.Button("Очистить диалог")

            self.send_button.click(
                self.send_message,
                inputs=[self.input_text, self.use_rag_checkbox],
                outputs=[self.chatbot, self.input_text, self.retrieved_ids_box, self.download_button]
            )

            self.clear_button.click(self.clear_chat, outputs=self.chatbot)
    
    def launch(self):
        """Запускает интерфейс."""
        self.app.launch()#share=True)

# Создание и запуск интерфейса
if __name__ == "__main__":
    chat_ui = ChatInterface(
        # chroma_config_path="config/embedding/e5-large-v2.yaml", 
        chroma_config_path="config/embedding/multilingual-e5-large.yaml", 

        # llm_config_path="config/LLM/qwen3-1.7b.yaml",
        # llm_config_path = "config/LLM/deepseek-r1-0528-qwen3-8b.yaml",
        # llm_config_path = "config/LLM/mistral-7b-instruct-v0.3.yaml",
        # llm_config_path = "config/LLM/qwen2.5-7b-instruct-1m.yaml",
        # llm_config_path = "config/LLM/llama-3.2-3b-instruct.yaml",
        llm_config_path = "config/LLM/qwen3-1.7b.yaml",

        # lang='en'
        lang='ru'
    )
    chat_ui.launch()
