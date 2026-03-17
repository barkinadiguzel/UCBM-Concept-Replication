import torch
from sklearn.decomposition import NMF

def compute_nmf(activations, n_concepts=50, init='random', max_iter=200):
    model = NMF(n_components=n_concepts, init=init, max_iter=max_iter, random_state=0)
    U = model.fit_transform(activations.numpy())
    C = model.components_.T  
    return torch.tensor(C, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
