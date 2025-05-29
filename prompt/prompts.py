class QuestionsToDoc():

    # Промпт для генерации вопросов
    # - Avoid yes/no questions—focus on "why," "how," "what are the implications," etc.
    QUESTION_FORMATION = """You're a candidate of mathematical sciences with expertise in scientific writing and comprehension.
    Your task is to generate two meaningful questions based on the provided scientific article. Each question must be semantically close to the content but not a direct quotation. The answers should be concise yet informative, directly addressing the questions with relevant details from the text.

    ### Requirements:
    - Questions should be analytical or explanatory, not just factual.
    - The questions should be in a human way.
    - Avoid "yes"/"no" questions in the answers — focus on the facts.
    - Answers should be brief (2-4 sentences) while covering the key information.
    - Ensure the questions highlight important aspects of the article.

    ### Input:
    **Document:**  
    {document}

    ### Output (IMPORTANT! Do not add comments, do not add additional markup.):
    {{
        "question_1": "Generated question based on the article.",
        "answer_1": "Generated answer, summarizing key details from the article.",
        "question_2": "Another generated question.",
        "answer_2": "Corresponding answer."
    }}
    
    BAD RESULT!
    ```json
        ### Output
    ```
    BAD RESULT!
    ```python
        ### Output
    ```
    """


    CORRECTING_DICKTIONARY = """Correct the following data structure to the correct dictionary given in the format below.

    ### Input:  
    {input}  


    ### Output (IMPORTANT! Do not add comments, do not add additional markup.):
    {{
        "question_1": "Generated question based on the article.",
        "answer_1": "Generated answer, summarizing key details from the article.",
        "question_2": "Another generated question.",
        "answer_2": "Corresponding answer."
    }}
    
    BAD RESULT!
    ```json
        ### Output
    ```
    BAD RESULT!
    ```python
        ### Output
    ```
    """

    CORRECTING_DICKTIONARY_ONE_QUESTION = """Correct the following data structure to the correct dictionary given in the format below.

    ### Input:  
    {input}  


    ### Output (IMPORTANT! Do not add comments, do not add additional markup.):
    {{
        "question_1": "Generated question based on the article.",
        "answer_1": "Generated answer, summarizing key details from the article.",
    }}
    
    BAD RESULT!
    ```json
        ### Output
    ```
    BAD RESULT!
    ```python
        ### Output
    ```
    """

    TRANSLATE_TO_RUSSIAN = """Вы кандидат математических наук с опытом в научном письме и понимании. Так же вы прекрасно владеете переводом научных текстов с английского на русский язык.
    
    Необходимо перевести на русский язык вопрос по документу. 
    В вопросе не ссылайся на статью. 
    Вопрос: {question}

    Необходимо перевести на русский язык ответ на вопрос по документу на основе контекста документа
    Ответ: {answer}

    Ты можешь использовать исходный документ для лучшего понимания контекста вопроса и ответа.
    ИСХОДНЫЙ ТЕКСТ (не переводи это): {doc}

    ВАЖНО! Не добавляй кооментарии, не добавляй разметку. 
    ОТВЕТЕ ИСПОЛЬЗУЙ ТОЛЬКО РУССКИЙ ЯЗЫК! Некоторые термины можно оставить на английском.
    ### Output :
    {{
        "question_1": "Вопрос на русском языке",
        "answer_1": "Ответ на русском языке",
    }}
    """
    

class RAG_context():
    """Промпты для настройки работы с контекстом RAG + LLM"""

    # Системный промпт для вопросно-ответной системы
    SYSTEM_QA = """It is necessary to clearly answer the user's question.  
    Use only the information presented in context. 
    Important!!!
    If there is no relevant information in the context, please reply: 
    There is no response.

    ### QUESTION 
    {question}

    ### CONTEXT 
    {context}
    """

    SYSTEM_QA_SHORT = """It is necessary to clearly answer the user's question.  
    If there is no relevant information in the context, please reply: 
    There is no response.

    ### QUESTION 
    {question}

    ### CONTEXT 
    {context}
    """