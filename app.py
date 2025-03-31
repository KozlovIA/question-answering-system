import gradio as gr
from source.chroma_manager import ChromaDBManager
from source.llm_manager import LMStudioClient


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
            context = ""
            system = ""
            if use_rag:
                system = "Answer the user's question using context"
                rag_results = self.chroma_db.query(query_text=message, n_results=2)
                retrieved_docs = rag_results.get("documents", [[]])[0]
                context = "\n".join(retrieved_docs)

            
            full_prompt = f"{system} \nUser: {message} \nContext: {context}"
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
        with gr.Blocks() as self.app:
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
        self.app.launch(share=True)

# Создание и запуск интерфейса
if __name__ == "__main__":
    chat_ui = ChatInterface(
        chroma_config_path="config/embedding/e5-large-v2.yaml", 
        llm_config_path="config/model.yaml"
    )
    chat_ui.launch()
