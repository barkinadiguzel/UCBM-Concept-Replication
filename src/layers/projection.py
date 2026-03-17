import torch
import torch.nn.functional as F

def cosine_projection(activation, concept_matrix):
    activation_norm = F.normalize(activation, dim=1)           
    concept_norm = F.normalize(concept_matrix.t(), dim=1)      
    return torch.matmul(activation_norm, concept_norm.t())    
