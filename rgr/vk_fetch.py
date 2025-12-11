import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_VERSION = "5.199"
VK_TOKEN = os.getenv("VK_TOKEN")

def fetch_posts_for_group(domain: str, count: int = 100, offset: int = 0):
    '''
    Делает запрос к VK API и возвращает посты со стены группы по её domain.
    Возвращает словарь с ключами: domain, post_id, text
    '''

    params = {
            "access_token": VK_TOKEN,
            "v": API_VERSION,
            "domain": domain,
            "count": count,
            "offset": offset,
            "filter": "all"        
            }
    
    resp = requests.get(
            "https://api.vk.com/method/wall.get",
            params=params
        ).json()
    
    data = resp.get("response")
    if not data:
        print("VK error:", resp)
        return result
    
    result = {
        "domain": [], "post_id": [], "text": []
    }
    
    for i in data.get("items", []):
        result["domain"].append(domain)
        result["post_id"].append(i["id"])
        result["text"].append(i.get("text", ""))

    return result

fetch_posts_for_group("poterjashkansk", count=100, offset=100)