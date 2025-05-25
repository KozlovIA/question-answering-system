# %%
from source.tinydb_manager import TinyDB_manager
from benchmark.LLM_benchmark import AnswerEvaluator
import pandas as pd
import os

# %%
# Список файлов для обработки
input_files = [
    "qwen2.5_7b_instruct.json",
    # "meta-llama-3.1-8b-instruct.json",
    # "gpt4chan-8b.json",
    "mistral-7b-instruct-v0.3.json",
    "qwen2.5-7b-instruct-1m.json",
    "llama-3.2-3b-instruct.json",
    "qwen3-1.7b.json",
    # "gemma-3-1b-it.yaml"
]

# %%
# Обработка каждого файла
for input_file in input_files:
    # Преобразование имени файла для использования в путях
    input_file_clean = input_file.replace('-', '_')

    # Путь к конфигурации TinyDB
    tiny_config = f"benchmark/output_LLM/{input_file_clean}"

    # Инициализация менеджера и оценщика
    tiny_manager = TinyDB_manager(tiny_config)
    bench = AnswerEvaluator(data=tiny_manager.export_json())
    metrics = bench.evaluate()

    # Объединение метрик и сохранение в Excel
    merged_dict = {**metrics[0], **metrics[2]}
    df = pd.DataFrame.from_dict(merged_dict, orient='index', columns=['score'])

    # Создание директории для вывода, если она не существует
    output_dir = 'benchmark/output_LLM'
    os.makedirs(output_dir, exist_ok=True)

    # Путь к выходному файлу
    output_file = os.path.join(output_dir, input_file_clean.replace('.json', '.xlsx'))

    # Сохранение DataFrame в Excel
    df.to_excel(output_file)

# %%
# --- Чтение всех Excel-файлов из папки и объединение ---

# Путь к директории с выходными файлами
output_dir = 'benchmark/output_LLM'

# Получение списка всех .xlsx файлов в директории
files = [f for f in os.listdir(output_dir) if f.endswith('.xlsx')]

# Загрузка каждого файла в DataFrame и переименование колонки
dfs = []
for file in files:
    filepath = os.path.join(output_dir, file)
    df = pd.read_excel(filepath, index_col=0)
    df.columns = [os.path.splitext(file)[0]]  # Переименование колонки в имя файла без расширения
    dfs.append(df)

# Объединение всех DataFrame по индексу (метрикам)
final_df = pd.concat(dfs, axis=1)

# Сохранение объединенного DataFrame в Excel
final_df.to_excel("benchmark/comparison_llm.xlsx")

# Вывод объединенного DataFrame
print(final_df)
