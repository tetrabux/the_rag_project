import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

app = FastAPI()
STATE = {}


@app.on_event("startup")
def startup():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    print("Model loaded.")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0


@app.post("/generate")
def generate(req: GenerateRequest):
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]

    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return {"text": result}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}
