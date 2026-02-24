from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

app = FastAPI()

loaded_models = {}

ALL_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def load_esm_model(model_name: str):
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
    return loaded_models[model_name]

@app.get("/api/predict")
def predict(sequence : str, position : int, mutation: str, model_name: str):
    tokenizer, model = load_esm_model(model_name)
    
    # Tokenize the input
    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position

    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    mutant_id = tokenizer.convert_tokens_to_ids(mutation)

    # Masks the original input
    inputs.input_ids[0, token_index_in_seq] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    position_logits = logits[0, token_index_in_seq]
    log_probs = torch.nn.functional.log_softmax(position_logits, dim=0)

    wildtype_log_prob = log_probs[wildtype_id].item()
    mutant_log_prob = log_probs[mutant_id].item()

    score = mutant_log_prob - wildtype_log_prob

    if score < -2.0:
        verdict = "Likely Damaging"
    elif score <= 0:
        verdict = "Likely Benign"
    else:
        verdict = "Uncertain / Neutral"

    print(verdict)

    return {"score": score, "verdict": verdict}


@app.get("/api/predict-all")
def predict_all(sequence: str, position: int, model_name: str):
    tokenizer, model = load_esm_model(model_name)
    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position

    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    wildtype_token = tokenizer.convert_ids_to_tokens(wildtype_id)

    # Masks the original character
    inputs.input_ids[0, token_index_in_seq] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    position_logits = logits[0, token_index_in_seq]
    log_probs = torch.nn.functional.log_softmax(position_logits, dim=0)
    wildtype_log_prob = log_probs[wildtype_id].item()

    results = []
    for aa in ALL_AMINO_ACIDS:
        mutant_id = tokenizer.convert_tokens_to_ids(aa)
        mutant_log_prob = log_probs[mutant_id].item()
        score = mutant_log_prob - wildtype_log_prob

        if score < -2.0:
            verdict = "Likely Damaging"
        elif score <= 0:
            verdict = "Likely Benign"
        else:
            verdict = "Uncertain / Neutral"

        results.append({
            "mutation": aa,
            "is_wildtype": aa == wildtype_token,
            "score": score,
            "verdict": verdict
        })

    # Sort best to worst score
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"wildtype": wildtype_token, "position": position, "results": results}


app.mount("/", StaticFiles(directory="static", html=True), name="static")