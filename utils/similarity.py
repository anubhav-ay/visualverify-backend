"""
Utility: Vector similarity functions.
"""

import numpy as np


def cosine_similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two numpy vectors.

    Args:
        vec_a: First vector (1-D).
        vec_b: Second vector (1-D).

    Returns:
        Float in range [0, 1]. Returns 0.0 on error.
    """
    try:
        vec_a = vec_a.flatten().astype(np.float32)
        vec_b = vec_b.flatten().astype(np.float32)

        # Pad shorter vector if shapes differ
        if vec_a.shape[0] != vec_b.shape[0]:
            max_len = max(vec_a.shape[0], vec_b.shape[0])
            vec_a = np.pad(vec_a, (0, max_len - vec_a.shape[0]))
            vec_b = np.pad(vec_b, (0, max_len - vec_b.shape[0]))

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clamp to [0, 1] — cosine can be negative for opposite vectors
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        print(f"[WARNING] cosine_similarity_score error: {e}")
        return 0.0
