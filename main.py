from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

app = FastAPI()

model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)


@app.get("/api/predict")
def predict(sequence : str, position : int, mutation: str):
    
    # Tokenize the input
    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position
    
    # Get the wild-type token ID and mutant token ID
    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    mutant_id = tokenizer.convert_tokens_to_ids(mutation)
    
    # Pass through model 
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Extract logits for the specific position
    position_logits = logits[0, token_index_in_seq]
    
    # Calculate Log-Likelihood Ratio (LLR)
    log_probs = torch.nn.functional.log_softmax(position_logits, dim=0)
    
    wildtype_log_prob = log_probs[wildtype_id].item()
    mutant_log_prob = log_probs[mutant_id].item()
    
    score = mutant_log_prob - wildtype_log_prob
    
    verdict = ""
    
    if score < -2.0:
        verdict = "Likely Damaging"
    elif score  <= 0:
        verdict = "Likely Benign"
    else:
        verdict = "Uncertain / Neutral"
        
    print(verdict)
    
    return {"score": score, "verdict": verdict}


app.mount("/", StaticFiles(directory="static", html=True), name="static")