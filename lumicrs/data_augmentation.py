# data_augmentation_paper43_multiLLM_online.py
# Paper 4.3: GPT-4o-Based Prototype Dialogue Data Augmentation (online multi-LLM voting)
#
# ===== Environment Variables (placeholders) =====
# OpenAI (also used for fallback votes):
#   export OPENAI_API_KEY="..."
#   export OPENAI_BASE_URL="https://api.openai.com/v1"     # optional (default shown)
#
# Claude3 (Anthropic):
#   export ANTHROPIC_API_KEY="..."
#   export ANTHROPIC_BASE_URL="https://api.anthropic.com"  # optional
#   export ANTHROPIC_VERSION="2023-06-01"                  # optional
#
# Gemini:
#   export GEMINI_API_KEY="..."
#   export GEMINI_BASE_URL="https://generativelanguage.googleapis.com"  # optional
#
# Grok (xAI, if OpenAI-compatible):
#   export XAI_API_KEY="..."
#   export XAI_BASE_URL="https://api.x.ai/v1"
#
# Llama (ONLINE, via OpenAI-compatible endpoint: Together/Fireworks/OpenRouter/self-hosted gateway etc.)
#   export LLAMA_API_KEY="..."             # sometimes optional
#   export LLAMA_BASE_URL="https://<your-openai-compatible-llama-endpoint>/v1"
#
# =================================================

import os
import json
import time
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import requests
from sentence_transformers import SentenceTransformer

# --- Optional SDK imports (requested); script still runs without them ---
# Gemini
try:
    import google.generativeai as genai  # noqa: F401
except Exception:
    genai = None

# Claude3
try:
    import anthropic  # noqa: F401
except Exception:
    anthropic = None

# OpenAI (also commonly used by OpenAI-compatible providers; Grok/Llama gateways may be compatible)
try:
    import openai  # noqa: F401
except Exception:
    openai = None


# =========================
# CONFIG (edit these two paths)
# =========================
INPUT_TRAIN_JSONL = "/media/wjz/2TBverydiao/ECR/src_emo/data/redial_gen/train_data_dbpedia_emo.jsonl"
OUTPUT_TRAIN_JSONL = "/media/wjz/2TBverydiao/ECR/src_emo/data/redial_gen/train_data_dbpedia_emo_aug.jsonl"

OUT_PROTOTYPES_JSON = "prototypes_43.json"
OUT_ACCEPTED_AUG_JSONL = "accepted_aug.jsonl"
OUT_HUMAN_REVIEW_JSONL = "human_review.jsonl"

# 4.3 settings
TEMP = 0.8
SIM_TH = 0.85
TAIL_GEN_RANGE = (8, 10)  # paper: 8–10
BODY_GEN_RANGE = (4, 5)   # paper: 4–5
CONTEXT_K = 3

# 4.2 popularity segmentation (paper example)
BODY_RANGE = (2, 5)
THETA = None  # None => first quartile

# FinalScore weights (normalized inside)
W_SEM, W_EMO, W_MOV, W_INT = 0.4, 0.2, 0.2, 0.2
ALPHA_OVERLAP = 1.0

# Embedding model for Step3 filtering + context retrieval
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

# Voting rule (paper: 5 LLMs)
TARGET_VOTES = 5
PASS_AUTO = 4          # >=4 accept
PASS_HUMAN_LOW = 2     # 2–3 human review
PASS_HUMAN_HIGH = 3

# Generation model/provider (online)
# Use OpenAI by default; you can switch to "llama" or "grok" if you want generation from them
GEN_PROVIDER = "openai"     # openai / llama / grok
GEN_MODEL = "gpt-4o"
GEN_MAX_TOKENS = 1800

# Voters: Gemini / Llama / Grok / Claude3 / OpenAI (OpenAI also used as fallback)
# If a voter isn't configured (missing key/base), it's skipped; missing votes are filled by OpenAI fallback.
JUDGE_VOTERS = [
    {"name": "gemini",   "type": "gemini",       "env_key": "GEMINI_API_KEY",    "env_base": "GEMINI_BASE_URL",    "default_base": "https://generativelanguage.googleapis.com", "model": "gemini-1.5-pro"},
    {"name": "llama",    "type": "openai_compat","env_key": "LLAMA_API_KEY",     "env_base": "LLAMA_BASE_URL",     "default_base": "",                                         "model": "llama-3.1-70b-instruct"},
    {"name": "grok",     "type": "openai_compat","env_key": "XAI_API_KEY",       "env_base": "XAI_BASE_URL",       "default_base": "https://api.x.ai/v1",                      "model": "grok-2"},
    {"name": "claude3",  "type": "anthropic",    "env_key": "ANTHROPIC_API_KEY", "env_base": "ANTHROPIC_BASE_URL", "default_base": "https://api.anthropic.com",                "model": "claude-3-5-sonnet-latest"},
    {"name": "openai",   "type": "openai_compat","env_key": "OPENAI_API_KEY",    "env_base": "OPENAI_BASE_URL",    "default_base": "https://api.openai.com/v1",                "model": "gpt-4o"},
]

# Speed: for context retrieval, take top-M by semantic then compute full score
SEM_CANDIDATES = 800

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMO_KEYS = ["happy", "curious", "like", "grateful", "neutral", "negative"]


# =========================
# Utils
# =========================
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def save_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Paper-aligned feature extraction
# =========================
def get_dialogue_text(sample: dict) -> str:
    msgs = sample.get("messages", [])
    texts = []
    for m in msgs:
        t = m.get("text", "")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return " [SEP] ".join(texts)

def get_movie_set(sample: dict) -> Set[str]:
    mm = sample.get("movieMentions", {})
    return set(mm.keys()) if isinstance(mm, dict) else set()

def get_movie_title(sample: dict, movie_id: str) -> str:
    mm = sample.get("movieMentions", {})
    if isinstance(mm, dict):
        return str(mm.get(movie_id, "")).strip()
    return ""

def get_emotion_vector(sample: dict) -> np.ndarray:
    msgs = sample.get("messages", [])
    if not msgs:
        return np.zeros(len(EMO_KEYS), dtype=np.float32)
    vals = []
    for m in msgs:
        emo = m.get("emotion", {})
        if not isinstance(emo, dict):
            emo = {}
        vals.append([float(emo.get(k, 0.0)) for k in EMO_KEYS])
    return np.mean(np.asarray(vals, dtype=np.float32), axis=0)


# =========================
# 4.2 prereq: popularity segmentation & prototypes
# =========================
def build_inverted_index(movie_sets: List[Set[str]]) -> Dict[str, List[int]]:
    inv = defaultdict(list)
    for i, ms in enumerate(movie_sets):
        for m in ms:
            inv[m].append(i)
    return inv

def segment_movies_by_popularity(inv: Dict[str, List[int]],
                                body_range: Tuple[int, int] = (2, 5),
                                theta: Optional[int] = None):
    pop = {m: len(ids) for m, ids in inv.items()}
    counts = np.asarray(list(pop.values()), dtype=np.float32)
    if theta is None:
        theta = int(np.quantile(counts, 0.25))  # paper example
    lo, hi = body_range
    M_tail = {m for m, c in pop.items() if c < theta}
    M_body = {m for m, c in pop.items() if lo <= c <= hi}
    M_head = set(pop.keys()) - M_tail - M_body
    return M_head, M_body, M_tail, pop

def select_prototype_for_movie(sample_ids: List[int], emb: torch.Tensor) -> int:
    # reproducible: highest semantic centrality
    if len(sample_ids) == 1:
        return sample_ids[0]
    idx = torch.tensor(sample_ids, dtype=torch.long, device=emb.device)
    e = emb.index_select(0, idx)
    c = F.normalize(e.mean(dim=0), dim=0)
    score = e @ c
    return sample_ids[int(torch.argmax(score).item())]


# =========================
# FinalScore (Eq.15-19) for selecting similar dialogues to build augmentation prompt
# =========================
def sim_sem_vec(emb_all: torch.Tensor, emb_c: torch.Tensor) -> torch.Tensor:
    return emb_all @ emb_c  # normalized dot = cosine

def sim_emo_vec(emo_all: torch.Tensor, emo_c: torch.Tensor) -> torch.Tensor:
    l1 = torch.sum(torch.abs(emo_all - emo_c), dim=-1)
    return 1.0 / (1.0 + l1)

def interaction_factor(sem: torch.Tensor, emo: torch.Tensor, mov: torch.Tensor, alpha: float) -> torch.Tensor:
    return sem * emo * torch.sigmoid(alpha * mov)

def final_score_from_parts(sem, emo, mov, R):
    s = W_SEM + W_EMO + W_MOV + W_INT
    w_sem, w_emo, w_mov, w_int = W_SEM/s, W_EMO/s, W_MOV/s, W_INT/s
    return w_sem*sem + w_emo*emo + w_mov*mov + w_int*R

def top_similar_dialogues_fast(
    proto_idx: int,
    emb: torch.Tensor,
    emo: torch.Tensor,
    movie_sets: List[Set[str]],
    topk: int,
    sem_candidates: int = 800
) -> List[int]:
    """
    Speed trick:
    1) take top-M by semantic similarity
    2) compute full FinalScore on those M only
    """
    N = emb.shape[0]
    emb_c = emb[proto_idx]
    emo_c = emo[proto_idx]
    proto_movies = movie_sets[proto_idx]

    sem_all = sim_sem_vec(emb, emb_c)
    sem_all[proto_idx] = -1e9
    M = min(sem_candidates, N-1)
    cand_idx = torch.topk(sem_all, k=M, largest=True).indices.detach().cpu().tolist()

    # compute emo & movie overlap on candidates
    cand_t = torch.tensor(cand_idx, dtype=torch.long, device=emb.device)
    sem = sem_all.index_select(0, cand_t)
    emo_s = sim_emo_vec(emo.index_select(0, cand_t), emo_c)

    # movie overlap (python set intersection on candidates)
    mov_np = np.array([len(movie_sets[i] & proto_movies) for i in cand_idx], dtype=np.float32)
    mov = torch.tensor(mov_np, dtype=torch.float32, device=emb.device)

    R = interaction_factor(sem, emo_s, mov, ALPHA_OVERLAP)
    score = final_score_from_parts(sem, emo_s, mov, R)

    k = min(topk, len(cand_idx))
    top_local = torch.topk(score, k=k, largest=True).indices.detach().cpu().tolist()
    return [cand_idx[i] for i in top_local]


# =========================
# Prompt build for 4.3 generation
# =========================
def dialogue_to_lines(sample: dict, max_turns: int = 10) -> str:
    msgs = sample.get("messages", [])
    lines = []
    for i, m in enumerate(msgs[:max_turns]):
        t = str(m.get("text", "")).strip()
        if not t:
            continue
        role = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{role}: {t}")
    return "\n".join(lines)

def build_generation_prompt(
    target_movie_id: str,
    target_movie_title: str,
    proto_sample: dict,
    context_samples: List[dict],
    n_dialogues: int
) -> List[Dict[str, str]]:
    movie_token = target_movie_id
    title_hint = f" (title: {target_movie_title})" if target_movie_title else ""

    examples = ["### Prototype Dialogue\n" + dialogue_to_lines(proto_sample)]
    for j, s in enumerate(context_samples):
        examples.append(f"### Similar Dialogue {j+1}\n" + dialogue_to_lines(s))
    examples_txt = "\n\n".join(examples)

    sys = (
        "You are a data augmentation assistant for a conversational movie recommender dataset.\n"
        "You MUST output valid JSON only. No markdown, no extra text.\n"
    )
    user = (
        f"Target movie token: {movie_token}{title_hint}\n\n"
        f"Using the dialogues below as style/content references, generate {n_dialogues} NEW multi-turn "
        f"recommendation dialogues.\n"
        f"Each dialogue MUST explicitly mention the target movie token exactly as given (do not alter it).\n"
        f"Dialogues should be diverse, natural, and plausible.\n\n"
        f"Output JSON format:\n"
        f'[\n  {{"dialogue":[{{"role":"user","text":"..."}},{{"role":"assistant","text":"..."}}, ... ]}},\n  ...\n]\n\n'
        f"Reference dialogues:\n{examples_txt}\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def parse_json_maybe(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
    return json.loads(t)

def flatten_dialogue(d: dict) -> str:
    turns = d.get("dialogue", [])
    parts = []
    for x in turns:
        parts.append(f'{x.get("role","")}: {x.get("text","")}')
    return "\n".join(parts).strip()

def contains_target_movie(d: dict, movie_token: str) -> bool:
    return movie_token in flatten_dialogue(d)


# =========================
# Step3: similarity filter
# =========================
@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum().item())


# =========================
# Online LLM calls: OpenAI-compatible / Anthropic / Gemini
# =========================
def voter_enabled(voter: dict) -> bool:
    base = _env(voter["env_base"], voter.get("default_base", ""))
    key = _env(voter["env_key"], "")
    if voter["type"] == "openai_compat":
        # allow keyless gateways; for ONLINE you usually have key, but base must exist
        return bool(base)
    return bool(base) and bool(key)

def openai_compat_chat(base_url: str, api_key: str, model: str,
                      messages: List[Dict[str, str]], temperature: float,
                      max_tokens: int, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def anthropic_chat(base_url: str, api_key: str, model: str,
                   messages: List[Dict[str, str]], temperature: float,
                   max_tokens: int, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": _env("ANTHROPIC_VERSION", "2023-06-01"),
    }

    system = ""
    user_parts = []
    for m in messages:
        if m["role"] == "system":
            system += (m["content"].strip() + "\\n")
        else:
            user_parts.append(f'{m["role"].upper()}: {m["content"]}')
    anth_messages = [{"role": "user", "content": "\\n".join(user_parts)}]

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system.strip(),
        "messages": anth_messages,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("content", "")
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    if isinstance(content, str):
        return content
    return ""

def gemini_chat(base_url: str, api_key: str, model: str,
                messages: List[Dict[str, str]], temperature: float,
                max_tokens: int, timeout: int = 60) -> str:
    url = base_url.rstrip("/") + f"/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = "\\n".join([f'{m["role"].upper()}: {m["content"]}' for m in messages])
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    cand = (data.get("candidates") or [{}])[0]
    content = cand.get("content") or {}
    parts = content.get("parts") or []
    return parts[0].get("text", "") if parts else ""

def llm_call(voter: dict, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    typ = voter["type"]
    base = _env(voter["env_base"], voter.get("default_base", ""))
    key = _env(voter["env_key"], "")
    model = voter["model"]

    # simple retry
    for attempt in range(3):
        try:
            if typ == "openai_compat":
                return openai_compat_chat(base, key, model, messages, temperature, max_tokens)
            if typ == "anthropic":
                return anthropic_chat(base, key, model, messages, temperature, max_tokens)
            if typ == "gemini":
                return gemini_chat(base, key, model, messages, temperature, max_tokens)
            raise ValueError(f"Unknown voter type: {typ}")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

def parse_pass_flag(text: str) -> bool:
    try:
        obj = parse_json_maybe(text)
        return bool(obj.get("pass", False)) if isinstance(obj, dict) else False
    except Exception:
        return False

def judge_messages(movie_token: str, proto_text: str, cand_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are a strict evaluator for augmented dialogues in a movie-recommendation dataset.\\n"
        "Return ONLY JSON: {\\"pass\\": true/false, \\"reason\\": \\"...\\"}\\n"
    )
    user = (
        f"Target movie token: {movie_token}\\n\\n"
        f"Prototype dialogue:\\n{proto_text}\\n\\n"
        f"Candidate dialogue:\\n{cand_text}\\n\\n"
        "Evaluate on: (1) semantic consistency with prototype topic, (2) fluency, "
        "(3) recommendation plausibility, and (4) the candidate clearly mentions the target movie token.\\n"
        "If the candidate is low-quality, off-topic, unnatural, or missing the token, set pass=false.\\n"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# =========================
# Fixed-5 voting with OpenAI补票 (marked)
# =========================
def openai_fallback_voter() -> dict:
    return {
        "name": "openai_fallback",
        "type": "openai_compat",
        "env_key": "OPENAI_API_KEY",
        "env_base": "OPENAI_BASE_URL",
        "default_base": "https://api.openai.com/v1",
        "model": "gpt-4o",
    }

def multi_llm_vote_fixed5(movie_token: str, proto_text: str, cand_text: str) -> Tuple[int, List[dict]]:
    msgs = judge_messages(movie_token, proto_text, cand_text)

    enabled = [v for v in JUDGE_VOTERS if voter_enabled(v)]
    details: List[dict] = []
    pass_cnt = 0

    # 1) real voters first
    for v in enabled[:TARGET_VOTES]:
        try:
            raw = llm_call(v, msgs, temperature=0.2, max_tokens=300)
            ok = parse_pass_flag(raw)
            details.append({"voter": v["name"], "model": v["model"], "pass": ok, "fallback": False})
            pass_cnt += int(ok)
        except Exception as e:
            details.append({"voter": v["name"], "model": v["model"], "pass": False, "fallback": False, "error": str(e)})

    # 2) fill remaining votes using OpenAI fallback
    remaining = TARGET_VOTES - len(details)
    if remaining > 0:
        fb = openai_fallback_voter()
        fb_ok = voter_enabled(fb)
        for k in range(remaining):
            if not fb_ok:
                details.append({
                    "voter": f"{fb['name']}#{k+1}",
                    "model": fb["model"],
                    "pass": False,
                    "fallback": True,
                    "source": "补票",
                    "error": "OpenAI fallback not configured. Set OPENAI_API_KEY and (optionally) OPENAI_BASE_URL."
                })
                continue

            try:
                raw = llm_call(fb, msgs, temperature=0.2, max_tokens=300)
                ok = parse_pass_flag(raw)
                details.append({
                    "voter": f"{fb['name']}#{k+1}",
                    "model": fb["model"],
                    "pass": ok,
                    "fallback": True,
                    "source": "补票"
                })
                pass_cnt += int(ok)
            except Exception as e:
                details.append({
                    "voter": f"{fb['name']}#{k+1}",
                    "model": fb["model"],
                    "pass": False,
                    "fallback": True,
                    "source": "补票",
                    "error": str(e)
                })

    # ensure exactly 5 vote records
    return pass_cnt, details[:TARGET_VOTES]


# =========================
# Build augmented record
# =========================
def build_aug_record(movie_id: str, movie_title: str, cand: dict, origin: str, uid: int) -> dict:
    # Keep schema robust: include emotion field as empty dict to avoid downstream key errors
    msgs = []
    for t in cand.get("dialogue", []):
        msgs.append({
            "text": t.get("text", ""),
            "role": t.get("role", "user"),
            "emotion": {}   # optional placeholder
        })
    return {
        "augmented": True,
        "aug_origin": origin,
        "aug_target_movie": movie_id,
        "movieMentions": {movie_id: movie_title} if movie_title else {movie_id: ""},
        "messages": msgs,
        "aug_id": f"aug_{origin}_{uid}",
    }


# =========================
# Generation call (online)
# =========================
def gen_call(provider: str, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    if provider == "openai":
        voter = {"type": "openai_compat", "env_key": "OPENAI_API_KEY", "env_base": "OPENAI_BASE_URL",
                 "default_base": "https://api.openai.com/v1", "model": model}
        return llm_call(voter, messages, temperature, max_tokens)
    if provider == "llama":
        voter = {"type": "openai_compat", "env_key": "LLAMA_API_KEY", "env_base": "LLAMA_BASE_URL",
                 "default_base": "", "model": model}
        return llm_call(voter, messages, temperature, max_tokens)
    if provider == "grok":
        voter = {"type": "openai_compat", "env_key": "XAI_API_KEY", "env_base": "XAI_BASE_URL",
                 "default_base": "https://api.x.ai/v1", "model": model}
        return llm_call(voter, messages, temperature, max_tokens)
    raise ValueError(f"Unknown GEN_PROVIDER: {provider}")


# =========================
# Main pipeline (paper 4.3)
# =========================
def main():
    if not os.path.exists(INPUT_TRAIN_JSONL):
        raise FileNotFoundError(f"train not found: {INPUT_TRAIN_JSONL}")

    device = pick_device()
    print(f"[Device] {device}")

    voters_on = [v["name"] for v in JUDGE_VOTERS if voter_enabled(v)]
    print(f"[Judge] enabled voters = {len(voters_on)}/{len(JUDGE_VOTERS)} -> {voters_on}")
    if len(voters_on) < TARGET_VOTES:
        print(f"[Judge] voters不足{TARGET_VOTES}个，将用 OpenAI 补票并标记 source=补票")

    # load train
    D = load_jsonl(INPUT_TRAIN_JSONL)
    print(f"[Load] train samples = {len(D)}")

    # embeddings
    embedder = SentenceTransformer(EMB_MODEL_NAME, device=device)

    texts = [get_dialogue_text(s) for s in D]
    emb_np = embedder.encode(texts, batch_size=256, show_progress_bar=True)
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device)
    emb = F.normalize(emb, dim=-1)

    emo_np = np.stack([get_emotion_vector(s) for s in D], axis=0).astype(np.float32)
    emo = torch.tensor(emo_np, dtype=torch.float32, device=device)

    movie_sets = [get_movie_set(s) for s in D]
    inv = build_inverted_index(movie_sets)

    # segmentation
    M_head, M_body, M_tail, pop = segment_movies_by_popularity(inv, BODY_RANGE, THETA)
    print(f"[Group] head={len(M_head)}, body={len(M_body)}, tail={len(M_tail)}")

    # prototypes for body∪tail
    prototypes: Dict[str, int] = {}
    for m in tqdm(list(M_body | M_tail), desc="Selecting prototypes"):
        ids = inv.get(m, [])
        if ids:
            prototypes[m] = select_prototype_for_movie(ids, emb)

    save_json(OUT_PROTOTYPES_JSON, prototypes)
    print(f"[Save] prototypes -> {OUT_PROTOTYPES_JSON} (count={len(prototypes)})")

    accepted_aug = []
    human_review = []
    uid = 0

    for m in tqdm(list(prototypes.keys()), desc="Augmenting movies (4.3)"):
        proto_idx = prototypes[m]
        proto_sample = D[proto_idx]
        proto_title = get_movie_title(proto_sample, m)

        origin = "tail" if m in M_tail else "body"
        n_gen = random.randint(*TAIL_GEN_RANGE) if origin == "tail" else random.randint(*BODY_GEN_RANGE)

        # Step1: context dialogues
        ctx_ids = top_similar_dialogues_fast(
            proto_idx=proto_idx,
            emb=emb,
            emo=emo,
            movie_sets=movie_sets,
            topk=CONTEXT_K,
            sem_candidates=SEM_CANDIDATES
        )
        context_samples = [D[i] for i in ctx_ids]

        # Step2: generate
        prompt_msgs = build_generation_prompt(m, proto_title, proto_sample, context_samples, n_dialogues=n_gen)
        try:
            raw = gen_call(GEN_PROVIDER, GEN_MODEL, prompt_msgs, temperature=TEMP, max_tokens=GEN_MAX_TOKENS)
        except Exception:
            continue

        try:
            cands = parse_json_maybe(raw)
            if not isinstance(cands, list):
                cands = []
        except Exception:
            cands = []

        if not cands:
            continue

        # Step3: similarity filter
        proto_text = get_dialogue_text(proto_sample)
        proto_emb = embedder.encode([proto_text], show_progress_bar=False)
        proto_emb_t = torch.tensor(proto_emb[0], dtype=torch.float32, device=device)

        for cand in cands:
            if not contains_target_movie(cand, m):
                continue

            cand_text = flatten_dialogue(cand)
            if not cand_text:
                continue

            cand_emb = embedder.encode([cand_text], show_progress_bar=False)
            cand_emb_t = torch.tensor(cand_emb[0], dtype=torch.float32, device=device)

            sim = cosine_sim(proto_emb_t, cand_emb_t)
            if sim > SIM_TH:
                continue

            # Step4: multi-LLM voting (fixed 5 with OpenAI补票)
            pass_cnt, vote_details = multi_llm_vote_fixed5(m, proto_text, cand_text)

            if pass_cnt >= PASS_AUTO:
                accepted_aug.append(build_aug_record(m, proto_title, cand, origin, uid))
                uid += 1
            elif PASS_HUMAN_LOW <= pass_cnt <= PASS_HUMAN_HIGH:
                human_review.append({
                    "movie_id": m,
                    "origin": origin,
                    "pass_votes": pass_cnt,
                    "sim_to_proto": sim,
                    "candidate": cand,
                    "vote_details": vote_details,
                })

    save_jsonl(OUT_ACCEPTED_AUG_JSONL, accepted_aug)
    save_jsonl(OUT_HUMAN_REVIEW_JSONL, human_review)
    print(f"[Save] accepted_aug -> {OUT_ACCEPTED_AUG_JSONL} (n={len(accepted_aug)})")
    print(f"[Save] human_review -> {OUT_HUMAN_REVIEW_JSONL} (n={len(human_review)})")

    expanded = D + accepted_aug
    save_jsonl(OUTPUT_TRAIN_JSONL, expanded)
    print(f"[Done] expanded train -> {OUTPUT_TRAIN_JSONL} (n={len(expanded)})")


if __name__ == "__main__":
    main()
