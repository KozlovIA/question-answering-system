class QuestionsToDoc():

    QUESTION_FORMATION = """You're a candidate of mathematical sciences with expertise in scientific writing and comprehension.
    Your task is to generate two meaningful questions based on the provided scientific article. Each question must be semantically close to the content but not a direct quotation. The answers should be concise yet informative, directly addressing the questions with relevant details from the text.

    ### Requirements:
    - Questions should be analytical or explanatory, not just factual.
    - Avoid yes/no questions—focus on "why," "how," "what are the implications," etc.
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
    