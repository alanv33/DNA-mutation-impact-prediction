from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import threading

loaded_models = {}
ALL_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def background_load():
    print("Starting background model loading...")
    load_esm_model("facebook/esm2_t6_8M_UR50D")
    print("Background loading complete.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=background_load, daemon=True).start()
    yield
    loaded_models.clear()

app = FastAPI(lifespan=lifespan)

def load_esm_model(model_name: str):
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        loaded_models[model_name] = (tokenizer, model)
    return loaded_models[model_name]

@app.get("/api/predict")
def predict(sequence: str, position: int, mutation: str, model_name: str):
    tokenizer, model = load_esm_model(model_name)

    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position

    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    mutant_id = tokenizer.convert_tokens_to_ids(mutation)

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

    return {"score": score, "verdict": verdict}


@app.get("/api/predict-all")
def predict_all(sequence: str, position: int, model_name: str):
    tokenizer, model = load_esm_model(model_name)
    inputs = tokenizer(sequence, return_tensors="pt")
    token_index_in_seq = position

    wildtype_id = inputs.input_ids[0, token_index_in_seq].item()
    wildtype_token = tokenizer.convert_ids_to_tokens(wildtype_id)

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

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"wildtype": wildtype_token, "position": position, "results": results}


@app.get("/api/scan")
def scan_sequence(sequence: str, model_name: str):
    """
    Runs the model ONCE for the full sequence, then scores all 20 amino acids
    at every position in a single forward pass. Returns crucialness data for
    each position based on how many AAs can replace it without being damaging.
    """
    tokenizer, model = load_esm_model(model_name)
    inputs = tokenizer(sequence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [1, seq_len+2, vocab_size]

    seq_len = len(sequence)
    positions = []

    for i, aa_char in enumerate(sequence):
        token_index = i + 1  # offset for [CLS] token

        wildtype_id = inputs.input_ids[0, token_index].item()
        wildtype_token = tokenizer.convert_ids_to_tokens(wildtype_id)

        position_logits = logits[0, token_index]
        log_probs = torch.nn.functional.log_softmax(position_logits, dim=0)
        wildtype_log_prob = log_probs[wildtype_id].item()

        tolerated = []
        damaging = []

        for aa in ALL_AMINO_ACIDS:
            if aa == wildtype_token:
                continue  # skip self-comparison
            mutant_id = tokenizer.convert_tokens_to_ids(aa)
            mutant_log_prob = log_probs[mutant_id].item()
            score = mutant_log_prob - wildtype_log_prob

            if score < -2.0:
                damaging.append(aa)
            else:
                tolerated.append(aa)

        tolerated_count = len(tolerated)  # out of 19 possible substitutions

        if tolerated_count <= 3:
            tier = "Core"
        elif tolerated_count <= 8:
            tier = "Important"
        elif tolerated_count <= 14:
            tier = "Flexible"
        else:
            tier = "Highly Flexible"

        positions.append({
            "position": i + 1,
            "residue": wildtype_token,
            "tolerated_count": tolerated_count,
            "damaging_count": len(damaging),
            "tolerated": tolerated,
            "damaging": damaging,
            "tier": tier
        })

    return {"sequence": sequence, "length": seq_len, "positions": positions}


@app.get("/api/run-tests")
def run_tests():
    """
    Runs pytest with -v --tb=short and parses the output directly.
    No extra plugins required.
    """
    import subprocess
    import os
    import re
    import time

    test_file = os.path.join(os.path.dirname(__file__), "test_mutations.py")
    start = time.time()

    result = subprocess.run(
        ["python", "-m", "pytest", test_file, "-v", "--tb=short", "-s"],
        capture_output=True, text=True
    )

    duration = round(time.time() - start, 2)
    output = result.stdout + result.stderr

    tests = []
    test_line_re = re.compile(
        r"test_mutations\.py::(\w+)::(\w+)\s+(PASSED|FAILED|ERROR)"
    )

    # Collect failure messages
    failure_blocks = {}
    fail_section = re.split(r"=+ FAILURES =+", output)
    if len(fail_section) > 1:
        blocks = re.split(r"_{5,}", fail_section[1])
        for block in blocks:
            lines = block.strip().splitlines()
            if not lines:
                continue
            name_match = re.search(r"(\w+)\s*$", lines[0])
            if name_match:
                failure_blocks[name_match.group(1)] = block.strip()

    # Split output into per-test chunks using PASSED/FAILED markers
    # Build a map of test_name -> its captured stdout block
    stdout_blocks = {}
    # pytest -s prints stdout inline; capture everything between test headers
    test_positions = [(m.start(), m.group(1), m.group(2), m.group(3))
                      for m in test_line_re.finditer(output)]

    for i, (pos, cls, tname, outcome_str) in enumerate(test_positions):
        # Look backward from the PASSED/FAILED line to find printed output
        start = test_positions[i-1][0] if i > 0 else 0
        chunk = output[start:pos]
        stdout_blocks[tname] = chunk.strip()

    for m in test_line_re.finditer(output):
        class_name = m.group(1)
        test_name  = m.group(2)
        outcome    = m.group(3).lower()

        # Extract LLR score from captured stdout
        chunk = stdout_blocks.get(test_name, "")
        score_match = re.search(r"Score:\s*([-\d.]+)", chunk)
        stdout_score = score_match.group(1) if score_match else ""

        # Build clean log lines for this test
        log_lines = []
        for line in chunk.splitlines():
            line = line.strip()
            if line and not line.startswith("PASSED") and not line.startswith("FAILED"):
                log_lines.append(line)

        tests.append({
            "id": f"test_mutations.py::{class_name}::{test_name}",
            "class": class_name,
            "name": test_name,
            "outcome": outcome,
            "duration": 0,
            "message": failure_blocks.get(test_name, ""),
            "stdout": stdout_score,
            "logs": "\n".join(log_lines)
        })

    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] in ("failed", "error"))

    return {
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "duration": duration,
        "tests": tests
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")