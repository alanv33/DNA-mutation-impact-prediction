import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import math

# Load the model and tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)

def score_mutation(sequence, position, mutant_aa):
    
    # 2. Tokenize the input
    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position + 1 
    
    # 3. Get the wild-type token ID and mutant token ID
    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    mutant_id = tokenizer.convert_tokens_to_ids(mutant_aa)
    
    # 4. Pass through model 
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 5. Extract logits for the specific position
    position_logits = logits[0, token_index_in_seq]
    
    # 6. Calculate Log-Likelihood Ratio (LLR)
    log_probs = torch.nn.functional.log_softmax(position_logits, dim=0)
    
    wildtype_log_prob = log_probs[wildtype_id].item()
    mutant_log_prob = log_probs[mutant_id].item()
    
    llr_score = mutant_log_prob - wildtype_log_prob
    
    return llr_score


wildtype_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

mutation_pos = 0 
target_aa = "P"

score = score_mutation(wildtype_seq, mutation_pos, target_aa)

print(f"Mutation: {wildtype_seq[mutation_pos]}{mutation_pos+1}{target_aa}")
print(f"ESM2 Score: {score:.4f}")

if score < -2.0:
    print("Prediction: Likely Deleterious")
elif score > 0:
    print("Prediction: Likely Benign / Beneficial")
else:
    print("Prediction: Uncertain / Neutral")