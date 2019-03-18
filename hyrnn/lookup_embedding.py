import torch
from torch.nn.modules.module import Module
import geoopt


class LookupEmbedding(Module):
    r"""A lookup table for embeddings, similar to :meth:`torch.nn.Embedding`,
    that replaces operations with their Poincare-ball counterparts.

    This module is intended to be used for word embeddings,
    retrieved by their indices.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim
        (int or tuple of ints): the shape of each embedding;
                                would've been better named embedding_shape,
                                if not for desirable name-level compatibility
                                with nn.Embedding;
                                embedding is commonly a vector,
                                but we do not impose such restriction
                                so as to not prohibit e.g. Stiefel embeddings.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, *embedding_dim).

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    def __init__(
        self, num_embeddings, embedding_dim, manifold=geoopt.Euclidean(), _weight=None
    ):
        super(LookupEmbedding, self).__init__()
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding_dim,)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        if _weight is None:
            _weight = torch.Tensor(num_embeddings, *embedding_dim)
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)
            self.reset_parameters()
        else:
            assert _weight.shape == (
                num_embeddings,
                *embedding_dim,
            ), "_weight MUST be of shape (num_embeddings, *embedding_dim)"
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)

    def reset_parameters(self):
        # TODO: allow some sort of InitPolicy
        #       as LookupEmbedding's parameter
        #       for e.g. random init;
        #       at the moment, you're supposed
        #       to do actual init on your own
        #       in the client code.
        with torch.no_grad():
            self.weight.fill_(0)

    def forward(self, input):
        shape = list(input.shape) + list(self.weight.shape[1:])
        shape = tuple(shape)
        return self.weight.index_select(0, input.reshape(-1)).view(shape)
