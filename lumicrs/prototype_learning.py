# prototype_learning_paper42_43.py
# Prototype Dialogue Selection (paper 4.2/4.3 prerequisite)
#
# This script:
# 1) Loads ReDial-style training JSONL
# 2) Computes dialogue-level embeddings (SentenceTransformer)
# 3) Computes movie popularity pop(m) from movieMentions
# 4) Splits movies into Head/Body/Tail using theta = 1st quartile (default) and body_range=(2,5)
# 5) For each movie in Body ∪ Tail, selects ONE prototype dialogue by "semantic centrality":
#       prototype = argmax_i cos(e_i, mean({e_j}))
#
# Output:
#   - prototypes_43.json: {movie_id: prototype_sample_index}
#   - prototypes_43_meta.json: additional stats (head/body/tail sizes, theta, pop histogram summary)

import os
import json
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# =========================
# 0) Config (edit paths)
# =========================
INPUT_TRAIN_JSONL = "/media/wjz/2TBverydiao/ECR/src_emo/data/redial_gen/train_data_dbpedia_emo.jsonl"
OUT_PROTOTYPES_JSON = "prototypes_43.json"
OUT_META_JSON = "prototypes_43_meta.json"

# Popularity segmentation (paper example)
BODY_RANGE = (2, 5)   # body: pop in [2,5]
THETA = None          # None => first quartile of pop values

# Embeddings
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================
# 1) IO
# =========================
def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 2) Feature extraction
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
    if isinstance(mm, dict):
        return set(mm.keys())
    return set()


# =========================
# 3) Popularity + inverted index
# =========================
def build_inverted_index(movie_sets: List[Set[str]]) -> Dict[str, List[int]]:
    inv = defaultdict(list)
    for i, ms in enumerate(movie_sets):
        for m in ms:
            inv[m].append(i)
    return inv

def segment_movies_by_popularity(
    inv: Dict[str, List[int]],
    body_range: Tuple[int, int] = (2, 5),
    theta: Optional[int] = None
):
    pop = {m: len(ids) for m, ids in inv.items()}
    counts = np.asarray(list(pop.values()), dtype=np.float32)

    if theta is None:
        theta = int(np.quantile(counts, 0.25))  # first quartile

    lo, hi = body_range
    M_tail = {m for m, c in pop.items() if c < theta}
    M_body = {m for m, c in pop.items() if lo <= c <= hi}
    M_head = set(pop.keys()) - M_tail - M_body
    return M_head, M_body, M_tail, pop, theta


# =========================
# 4) Prototype selection (semantic centrality)
# =========================
def select_prototype_for_movie(sample_ids: List[int], emb: torch.Tensor) -> int:
    """
    Given indices of dialogues containing a movie m, select one prototype dialogue.
    Strategy: max cosine similarity to the centroid embedding of that movie's dialogues.
    """
    if len(sample_ids) == 1:
        return sample_ids[0]

    idx = torch.tensor(sample_ids, dtype=torch.long, device=emb.device)
    e = emb.index_select(0, idx)          # (K,d)
    c = F.normalize(e.mean(dim=0), dim=0) # centroid
    score = e @ c                        # (K,)
    best_local = int(torch.argmax(score).item())
    return sample_ids[best_local]


# =========================
# 5) Main
# =========================
def main():
    if not os.path.exists(INPUT_TRAIN_JSONL):
        raise FileNotFoundError(f"train not found: {INPUT_TRAIN_JSONL}")

    D = load_jsonl(INPUT_TRAIN_JSONL)
    print(f"[Load] train samples = {len(D)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Compute dialogue embeddings
    embedder = SentenceTransformer(EMB_MODEL_NAME, device=device)
    texts = [get_dialogue_text(s) for s in D]
    emb_np = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device)
    emb = F.normalize(emb, dim=-1)

    # Movie sets + inverted index
    movie_sets = [get_movie_set(s) for s in D]
    inv = build_inverted_index(movie_sets)

    # Popularity segmentation
    M_head, M_body, M_tail, pop, theta = segment_movies_by_popularity(inv, BODY_RANGE, THETA)
    print(f"[Group] head={len(M_head)}, body={len(M_body)}, tail={len(M_tail)} (theta={theta}, body_range={BODY_RANGE})")

    # Select prototypes for Body ∪ Tail
    targets = list(M_body | M_tail)
    prototypes: Dict[str, int] = {}

    for m in tqdm(targets, desc="Selecting prototypes (Body∪Tail)"):
        ids = inv.get(m, [])
        if not ids:
            continue
        prototypes[m] = select_prototype_for_movie(ids, emb)

    save_json(OUT_PROTOTYPES_JSON, prototypes)

    # Meta for debugging / paper traceability
    pop_values = np.asarray(list(pop.values()), dtype=np.int32)
    meta = {
        "input_train_jsonl": INPUT_TRAIN_JSONL,
        "embedding_model": EMB_MODEL_NAME,
        "device": device,
        "theta": int(theta),
        "body_range": list(BODY_RANGE),
        "num_movies_total": int(len(pop)),
        "num_movies_head": int(len(M_head)),
        "num_movies_body": int(len(M_body)),
        "num_movies_tail": int(len(M_tail)),
        "num_prototypes_saved": int(len(prototypes)),
        "pop_min": int(pop_values.min()) if pop_values.size else 0,
        "pop_max": int(pop_values.max()) if pop_values.size else 0,
        "pop_mean": float(pop_values.mean()) if pop_values.size else 0.0,
        "pop_q25": float(np.quantile(pop_values, 0.25)) if pop_values.size else 0.0,
        "pop_q50": float(np.quantile(pop_values, 0.50)) if pop_values.size else 0.0,
        "pop_q75": float(np.quantile(pop_values, 0.75)) if pop_values.size else 0.0,
    }
    save_json(OUT_META_JSON, meta)

    print(f"[Save] prototypes -> {OUT_PROTOTYPES_JSON} (count={len(prototypes)})")
    print(f"[Save] meta -> {OUT_META_JSON}")


if __name__ == "__main__":
    main()
