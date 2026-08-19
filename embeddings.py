import os
import re
import hashlib
import numpy as np
import config

def get_embedding(text: str) -> list[float]:
    """
    Generates a fast, stable, deterministic 1536-dimensional float32 vector
    using SHA256 word-bag and 3-gram character subword hashing.
    Runs locally in 0.1ms with zero network latency.
    """
    if not text:
        return [0.0] * config.EMBEDDING_DIM

    norm = text.lower().strip()
    words = re.findall(r'\w+', norm)
    dim = config.EMBEDDING_DIM
    vec = np.zeros(dim, dtype=np.float32)

    for word in words:
        # Full word hash (weight = 2.0)
        h_full = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % dim
        vec[h_full] += 2.0

        # Subword 3-gram character hashes (weight = 1.0) for typo resilience
        for i in range(len(word) - 2):
            ngram = word[i:i+3]
            h_ng = int(hashlib.sha256(ngram.encode('utf-8')).hexdigest(), 16) % dim
            vec[h_ng] += 1.0

    norm_val = np.linalg.norm(vec)
    if norm_val > 0:
        vec = vec / norm_val

    return vec.tolist()

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculates cosine similarity between two float vectors.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        return float(dot / (norm1 * norm2))
    return 0.0
