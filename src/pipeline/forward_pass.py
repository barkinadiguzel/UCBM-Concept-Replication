import torch
from layers.backbone import Backbone
from layers.projection import cosine_projection
from layers.gating import Gating
from layers.linear import SparseLinear
from concept.concept_builder import build_concepts

x_demo = torch.randn(1, 3, 224, 224)  # dummy input

backbone = Backbone()
A = backbone(x_demo)  # [1, 2048]

C = build_concepts(A, n_concepts=10) 

proj = cosine_projection(A, C)  # [1, k]

gate = Gating(offset=0.1)
pi = gate(proj) 

linear = SparseLinear(input_dim=pi.size(1), output_dim=5)
out = linear(pi)

print("Concept projections:", pi)
print("Classifier output:", out)
