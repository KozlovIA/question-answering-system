from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
import yaml

class LMStudioClient:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.llm = ChatOpenAI(
            openai_api_base=self.config.get("api_url", "http://localhost:1234/v1"),
            openai_api_key="lm-studio",  # API-ключ не проверяется, можно любое значение
            # model_name=self.config.get("model_name", "qwen2.5-7b-instruct"),
            temperature=self.config.get("temperature", 0.8),
            max_tokens=self.config.get("max_tokens", 32768),
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template("{query}")]
        )
        self.context = []

    @staticmethod
    def load_config(file_path: str):
        """Считывает YAML-файл конфигурации."""
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def update_context(self, user_input: str):
        """Добавляет сообщение пользователя в контекст."""
        self.context.append(HumanMessage(content=user_input))

    def clear_context(self):
        """Очищает контекст общения."""
        self.context = []

    def post_completion(self, user_input: str) -> str:
        """Отправляет запрос к модели и получает ответ."""
        self.update_context(user_input)
        response = self.llm.invoke(self.context)
        self.context.append(response)  # Добавляем ответ в контекст
        return response.content

if __name__ == "__main__":
    CONFIG_PATH = "config/model.yaml"
    client = LMStudioClient(CONFIG_PATH)
    
    user_input = "Who created you?"
    response = client.post_completion(user_input)
    print(response)
    
    # Очистка контекста
    client.clear_context()
