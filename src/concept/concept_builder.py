import torch
from .nmf import compute_nmf

def build_concepts(activations, n_concepts=50):
    C, _ = compute_nmf(activations, n_concepts=n_concepts)
    return C
