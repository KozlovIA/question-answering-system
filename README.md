# Вопросы к документам

Хранятся в коллекции chroma `question_to_doc.py`, формируются при помощи `qwen2.5-7b-instruct-1m`

# Модели эмбеддингов для тестирования

✔ **Если нужно быстро и эффективно** → `all-MiniLM-L6-v2`

✔ **Если приоритет — точность** → `all-mpnet-base-v2`

✔ **Если важен русский язык** → `paraphrase-multilingual-MiniLM-L12-v2`

✔ **Если делаешь RAG для QA** → `multi-qa-mpnet-base-dot-v1`

# Создание БД

`python doc_to_chroma.py [config_name]`, однако лучше использовать `doc_to_chroma.ipynb`, более новый + записывает по несколько моделей за раз и удаляет файлы не подходящие по числу токенов

# Генерация вопросов и эталонных ответов

`question_gen.py`

### **Как скачать модели вручную и использовать в LM Studio**

1. **Иди на Hugging Face** → [https://huggingface.co/TheBloke]()

   Это репозиторий TheBloke, который содержит большинство моделей в формате GGUF.
2. **Найди нужную модель** (я уже подготовил ссылки на Q5_K_M-версии):

   * **Mistral 7B Instruct** : [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF]()
   * **Mixtral 8x7B Instruct** : [https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF]()
   * **Nous Hermes 2 Mistral 7B** : [https://huggingface.co/TheBloke/Nous-Hermes-2-Mistral-GGUF]()
   * **OpenHermes 2.5 Mistral 7B** : [https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-GGUF]()
   * **Zephyr 7B Beta** : [https://huggingface.co/TheBloke/Zephyr-7B-Beta-GGUF]()
   * **GPT-4-judging-lmsys** : [https://huggingface.co/TheBloke/gpt-4-judging-lmsys-GGUF]()


# Для оценки QA-моделей используются метрики:

* **EM (Exact Match)** – доля полностью правильных ответов.
* **F1-score** – баланс точности (precision) и полноты (recall).
* **BLEU / ROUGE / METEOR** – если ответы генерируются (а не выбираются из текста).
* **Mean Reciprocal Rank (MRR)** – учитывает ранжирование правильного ответа.
