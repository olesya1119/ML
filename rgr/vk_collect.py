from vk_fetch import fetch_posts_for_group
import pandas as pd
import time
from pathlib import Path

# Список групп для сбора постов
GROUPS = [
    "poterjashkansk",
    "pet911nsk",
    "poiskvoronej",
    "kyterg_msk",
    "zhivotnye_darom"
]

SLEEP_SEC = 0.5 # VK запрещает чаще 3 запросов в секунду
POSTS_PER_GROUP = 1000
MAX_POSTS_COUNT = 100


def collect_posts(groups=GROUPS, posts_per_group=POSTS_PER_GROUP, sleep_sec=SLEEP_SEC):
    """
    Собирает посты из нескольких групп.
    groups - список доменов групп ВК.
    posts_per_group — сколько постов на группу (максимум, зависит от стены).
    """
    all_rows = {"domain": [], "post_id": [], "text": []}

    for domain in groups:
        print(f"Собираю группу {domain}...")
        for offset in range(0, posts_per_group, MAX_POSTS_COUNT):
            chunk = fetch_posts_for_group(domain, count=MAX_POSTS_COUNT, offset=offset)
            if not chunk["post_id"]:
                break

            all_rows["domain"].extend(chunk["domain"])
            all_rows["post_id"].extend(chunk["post_id"])
            all_rows["text"].extend(chunk["text"])

            time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)

    before = len(df)
    df = df.drop_duplicates(subset=["domain", "post_id"], keep="first").reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Удалено дубликатов: {before - after}")

    return df


def collect_and_save_to_csv(output_path):
    """
    Собирает посты из указанных групп и сохраняет результат в CSV.
    """
    df = collect_posts()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Сохранено {len(df)} постов в файл: {output_path}")

    return df


if __name__ == "__main__":
    df_raw = collect_and_save_to_csv("rgr/data/vk_posts_raw.csv")

