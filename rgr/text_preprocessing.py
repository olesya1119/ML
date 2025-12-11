import re
from typing import Dict, Any

import pandas as pd
from transformers import AutoTokenizer

# Модель для русско-/мультиязычного текста
# при желании можно заменить, например, на "cointegrated/rubert-tiny2"
MODEL_NAME = "xlm-roberta-base"

# Загружаем токенизатор один раз при импорте модуля
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text: str) -> str:
    """
    Базовая очистка текста:
    - приведение к нижнему регистру
    - удаление ссылок, хэштегов, упоминаний вида [id123|Имя]
    - удаление лишних символов
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # ссылки
    text = re.sub(r"http\S+|www\S+", " ", text)

    # упоминания вида [id123|Имя]
    text = re.sub(r"\[id\d+\|[^\]]+\]", " ", text)

    # хэштеги
    text = re.sub(r"#\S+", " ", text)

    # всё, что не буквы/пробелы
    text = re.sub(r"[^а-яёa-z\s]", " ", text)

    # лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_with_transformer(
    text: str,
    max_length: int = 128,
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
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
        max_length=max_length,
        add_special_tokens=add_special_tokens,
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


def add_transformer_tokens_column(
    df: pd.DataFrame,
    text_column: str = "text",
    max_length: int = 128,
    new_column: str = "transformer_tokens",
) -> pd.DataFrame:
    """
    Добавляет в DataFrame колонку с токенами Transformers.

    :param df: исходный DataFrame
    :param text_column: колонка с текстом (например, 'text' из VK)
    :param max_length: максимальная длина последовательности
    :param new_column: имя новой колонки для токенов
    :return: копия DataFrame с добавленной колонкой
    """
    df = df.copy()

    def _tokenize_row(text: str):
        res = tokenize_with_transformer(text, max_length=max_length)
        # Можно сохранять только токены, чтобы удобнее смотреть
        return res["tokens"]

    df[new_column] = df[text_column].fillna("").apply(_tokenize_row)
    return df


if __name__ == "__main__":
    # Пример локального теста модуля
    example_text = "Нашёлся котёнок возле ТЦ, писать в ЛС! #найден #котик"
    res = tokenize_with_transformer(example_text)
    print("Очищенный текст:", clean_text(example_text))
    print("Токены:", res["tokens"])
    print("input_ids (первые 10):", res["input_ids"][:10])
