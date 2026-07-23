import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

JUDGE_CACHE_PATH = Path(__file__).parent / ".cache" / "judgments.json"
JUDGE_MODEL = "poolside/laguna-m.1:free"  # deliberately different from the generation model, to avoid self-grading bias

def make_client():
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def build_judge_prompt(query, answer, results):
    context = [f"{i+1} {chunk.text}" for i,(chunk,_) in enumerate(results)]
    context = "\n\n".join(context)

    instruction = """You are grading an AI-generated answer against the retrieved context it was given.

        Task 1: Break the ANSWER down into individual factual claims. For each claim, decide whether it is directly supported by the numbered CONTEXT below. A claim is "supported" only if the context actually states it — do not use outside knowledge.
       
        Task 2: Judge whether the ANSWER addresses the QUESTION. Pick exactly one verdict: "fully_addresses", "partially_addresses", "does_not_address", or "evades".
        
        Respond with ONLY a raw JSON object, no markdown code fences, no extra text, in exactly this shape:
        {"claims": [{"claim": "...", "supported": true, "reasoning": "..."}], "relevance": {"verdict": "...", "reasoning": "..."}}
        """
    return f"{instruction} \n\n {context} \n\n Question : {query} \n\n Answer : {answer}"

def judge_cache_key(query, answer, results):
    chunks = []
    for chunk,score in results:
        chunks.append(f"{chunk.file_path} : {chunk.text}")
    
    key_base = query + "\n" + JUDGE_MODEL + "\n" + "\n".join(chunks) + "\n" + answer
    
    hashed = hashlib.sha256(key_base.encode('utf-8')).hexdigest()

    return hashed


def load_judge_cache():
    if not JUDGE_CACHE_PATH.exists():
        return {}
    with open(JUDGE_CACHE_PATH, "r") as f:
        return json.load(f)


def save_judge_cache(cache):
    JUDGE_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(JUDGE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def parse_judge_response(text):
    text = text.strip()

    # models sometimes wrap the JSON in a markdown fence despite being told not to
    if text.startswith("```"):
       text = text.strip("`")
       if text.startswith("json"):
           text = text[4:]

       text = text.strip()
    
    return json.loads(text)

def compute_faithfulness(claims):
    if not claims:
        return 1.0  # no claims made means nothing to be unfaithful about
    supported = sum(1 for c in claims if c["supported"])
    return supported / len(claims)


def judge_answer(query, answer, results, cache, client):
    key_base = judge_cache_key(query, answer, results)

    if key_base in cache:
        return cache[key_base]
   
    prompt = build_judge_prompt(query, answer, results)

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=4096,  # this is a reasoning model — it can burn most of a small budget on internal reasoning before writing the JSON
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if not resp.choices:
            raise ValueError(f"empty response from provider: {resp}")

        content = resp.choices[0].message.content
        if content is None:
            raise ValueError(f"empty message content from provider: {resp}")

        parsed = parse_judge_response(content)

        judgement = {
            "faithfulness" : compute_faithfulness(parsed["claims"]),
            "relevance" : parsed["relevance"]["verdict"],
            "claims" : parsed["claims"],
            "relevance_reasoning" : parsed["relevance"]["reasoning"]
        }

        cache[key_base] = judgement
        
        save_judge_cache(cache)

        return judgement

    except Exception as e:
        # not cached, same as generation.py — a rate-limit or parse failure should be retried on rerun, not remembered
        print(f"Error judging answer: {e}")
        return {"faithfulness": None, "relevance": None, "claims": [], "relevance_reasoning": None}
        




    
