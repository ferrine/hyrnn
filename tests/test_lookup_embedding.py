from hyrnn.lookup_embedding import LookupEmbedding
import torch
from geoopt import PoincareBall


def test_lookup_embedding_construction():
    torch.manual_seed(42)
    num_embeddings = 4
    embedding_dim = 10
    W = torch.randn(num_embeddings, embedding_dim)
    table = LookupEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        _weight=W,
        manifold=PoincareBall(),
    )
    pred = table.forward(torch.tensor([0]))
    expected = W.index_select(0, torch.tensor([0]))
    assert pred.shape == expected.shape
    abs_error = (pred - expected).abs().sum().item()
    assert abs_error < 1e-9
