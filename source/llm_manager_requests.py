import requests

class LMStudioClient:
    def __init__(self, api_url, temperature=0.8, top_k=35, top_p=0.95, n_predict=400, max_tokens=None, stop=None):
        self.api_url = api_url
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_predict = n_predict
        self.max_tokens = max_tokens
        self.stop = stop or ["</s>", "Assistant:", "User:"]
        self.context = ""
        
    def update_context(self, user_input):
        """Обновляет контекст с последним вводом пользователя"""
        self.context += f"User: {user_input}\nAssistant:"
        
    def clear_context(self):
        """Очищает контекст общения"""
        self.context = ""
        
    def post_completion(self, user_input):
        """Отправляет запрос на получение ответа от модели"""
        self.update_context(user_input)
        data = {
            'prompt': self.context,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'n_predict': self.n_predict,
            'max_tokens': self.max_tokens,
            'stop': self.stop
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()  # Проверка на ошибки HTTP
            result = response.json()
            answer = result.get("choices", [{}])[0].get("text", "").strip()
            self.context += f" {answer}\n"  # Обновляем контекст с ответом ассистента
            return answer
        except requests.exceptions.RequestException as e:
            return f"Error processing your request: {e}"


if __name__ == "__main__":
    # Пример использования
    api_url = "http://localhost:1234/v1/completions"  # Убедись, что указал правильный URL
    import yaml 
    def load_from_yaml(file_path: str):
        """Считывает YAML файл и возвращает список тем."""
        with open(file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data
    
    MODEL_CONFIG = load_from_yaml("config/model.yaml")

    lm_client = LMStudioClient(**MODEL_CONFIG)

    user_input = "Who created you?"
    response = lm_client.post_completion(user_input)
    print(response)

    # user_input = "Can you explain how a black hole works?"
    # response = lm_client.post_completion(user_input)
    # print(response)

    # Очистить контекст, если нужно
    lm_client.clear_context()
