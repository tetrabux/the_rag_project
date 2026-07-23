import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CACHE_PATH = Path(__file__).parent / ".cache" / "generations.json"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def make_client():
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


def save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def build_prompt(query, results):
    context = [f"{i+1} {chunk.text}" for i,(chunk,_) in enumerate(results)]
    context = "\n\n".join(context)
    # explicit "say so" instruction discourages guessing when context is thin — makes faithfulness failures
    # look like hallucination, not an honest "I don't know" getting misjudged
    instruction = f" answer the question using only the numbered context below; if the context doesn't contain the answer, say so explicitly instead of guessing. \n\n {context} \n\n Question : {query}"

    return instruction



def cache_key(query, results):
    chunks = []
    for chunk,score in results:
        chunks.append(f"{chunk.file_path} : {chunk.text}")
    
    key_base = query + "\n" +MODEL + "\n" + "\n".join(chunks)
    
    hashed = hashlib.sha256(key_base.encode('utf-8')).hexdigest()

    return hashed


def generate_answer(query, results, cache, client):
    key_base = cache_key(query, results)
    if key_base in cache:
        return {"answer": cache[key_base], "cached": True}
    
    prompt = build_prompt(query, results)
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if not resp.choices:
            raise ValueError(f"empty response from provider: {resp}")
        answer = resp.choices[0].message.content
        cache[key_base] = answer

        save_cache(cache)
        return {"answer": answer, "cached": False}
    except Exception as e:
        # not cached on purpose — a rerun should retry, not permanently remember a rate-limit as "the answer"
        print(f"Error generating answer: {e}")
        return {"answer": None, "cached": False}


    
