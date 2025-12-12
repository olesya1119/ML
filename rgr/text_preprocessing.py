import re
from typing import Dict, Any

import pandas as pd
from transformers import AutoTokenizer

# Модель для русско-/мультиязычного текста
# при желании можно заменить, например, на "cointegrated/rubert-tiny2"
MODEL_NAME = "xlm-roberta-base"

# Загружаем токенизатор один раз при импорте модуля
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


import re

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]+")


def clean_text(text):
    """
    Очистка текста:
    """
    # Нормализуем пробелы/неразрывные пробелы
    text = text.replace("\xa0", " ").strip()

    # Убираем VK-упоминания вида [id123|Имя Фамилия]
    text = re.sub(r"\[id\d+\|[^\]]+\]", " ", text)

    # Удаляем ссылки
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Удаляем телефоны: +7 913 480-31-60, 8(913)123-45-67, 89131234567 и т.п.
    text = re.sub(r"\+?\d[\d\-\s\(\)]{7,}\d", " ", text)

    # Удаляем хештеги целиком (#потеряшканск)
    text = re.sub(r"#\S+", " ", text)

    # Удаляем эмодзи
    text = EMOJI_PATTERN.sub(" ", text)

    # Приводим к нижнему регистру
    text = text.lower()

    # Оставляем буквы, цифры и базовую пунктуацию; всё остальное выкидываем
    text = re.sub(r"[^a-zа-яё0-9\.\,\!\?\s]", " ", text)

    # Схлопываем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return text



def tokenize_with_transformer(text):
    """
    Токенизация одной строки текста с помощью токенизатора Transformers.
    Возвращает словарь с:
    - input_ids: список id токенов
    - attention_mask: маска (1 — реальный токен, 0 — паддинг)
    - tokens: список строковых токенов
    """
    text = clean_text(text)

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
    }


def read_csv_and_create_tokens_dataframe(filename):
    """
    Создает DataFrame с токенами Transformers из прочитанного csv-файла.
    """
    df = pd.read_csv(filename)

    def _tokenize_row(text: str):
        res = tokenize_with_transformer(text)
        return res["tokens"]

    df["transformer_tokens"] = df["text"].fillna("").apply(_tokenize_row)
    return df


if __name__ == "__main__":
    # Пример локального теста модуля
    example_text = "Нашёлся котёнок возле ТЦ, писать в ЛС! #найден #котик"
    res = tokenize_with_transformer(example_text)
    print("Очищенный текст:", clean_text(example_text))
    print("Токены:", res["tokens"])
    print("input_ids (первые 10):", res["input_ids"][:10])

    print(read_csv_and_create_tokens_dataframe("rgr/data/vk_posts_raw.csv"))
