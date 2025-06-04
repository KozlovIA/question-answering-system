import gradio as gr
from source.chroma_manager import ChromaDBManager
from source.llm_manager import LMStudioClient
from prompt.prompts import QA_context

from gradio.themes.utils import colors
from gradio.themes.base import Base

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


class ChatInterface:
    def __init__(self, chroma_config_path: str, llm_config_path: str):
        self.chat_history = []  # История сообщений
        self.chroma_db = ChromaDBManager(config_path=chroma_config_path)
        self.llm_client = LMStudioClient(config_path=llm_config_path)
        self.setup_interface()
    
    def send_message(self, message, use_rag):
        """
        Обрабатывает сообщение, используя LLM и (опционально) RAG.
        :param message: Введённое пользователем сообщение.
        :param use_rag: Флаг использования RAG.
        :return: Ответ модели и обновлённую историю чата.
        """
        try:
            system = QA_context.SYSTEM_QA_WITHOUT_RAG.format(question=message)
            if use_rag:
                
                rag_results = self.chroma_db.query(query_text=message, n_results=2)
                retrieved_docs = rag_results.get("documents", [[]])[0]
                context = "\n".join(retrieved_docs)
                
                system = QA_context.SYSTEM_QA_SHORT_RUS.format(question=message, context=context)
            
            full_prompt = system
            response = self.llm_client.post_completion(full_prompt)
            self.chat_history.append((message, response))
        except Exception as e:
            response = f"Произошла ошибка: {str(e)}"
            self.chat_history.append((message, response))
        
        return self.chat_history, ""
    
    def clear_chat(self):
        """Очищает историю чата."""
        self.chat_history = []
        self.llm_client.clear_context()
        return self.chat_history
    
    def setup_interface(self):
        """Настраивает интерфейс Gradio."""
        with gr.Blocks(theme=LightTheme()) as self.app:      # theme=gr.themes.Default()
            self.chatbot = gr.Chatbot(height=800)
            self.input_text = gr.Textbox(label="Введите сообщение")
            self.use_rag_checkbox = gr.Checkbox(label="Использовать RAG", value=False)
            
            with gr.Row():
                self.send_button = gr.Button("Отправить")
                self.clear_button = gr.Button("Очистить диалог")
            
            self.send_button.click(self.send_message, 
                                  inputs=[self.input_text, self.use_rag_checkbox], 
                                  outputs=[self.chatbot, self.input_text])
            self.clear_button.click(self.clear_chat, outputs=self.chatbot)
    
    def launch(self):
        """Запускает интерфейс."""
        self.app.launch()#share=True)

# Создание и запуск интерфейса
if __name__ == "__main__":
    chat_ui = ChatInterface(
        # chroma_config_path="config/embedding/e5-large-v2.yaml", 
        chroma_config_path="config/embedding/multilingual-e5-large.yaml", 
        llm_config_path="config/LLM/qwen3-1.7b.yaml"
    )
    chat_ui.launch()
