import torch
from .module import Module
from torch.nn import functional as F
from torch.nn import init
from torch._jit_internal import weak_module, weak_script, weak_script_method


@weak_module
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
    def __init__(self, num_embeddings, embedding_dim, manifold, _weight=None):
        super(LookupEmbedding, self).__init__()
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding,)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if _weight is None:
            _weight = torch.Tensor(num_embeddings, *embedding_dim)
        assert self.weight.shape == (num_embeddings, *embedding_dim), \
               '_weight MUST be of shape (num_embeddings, *embedding_dim)'
        self.weight = ManifoldParameter(_weight, manifold=manifold)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: allow some sort of InitPolicy
        #       as LookupEmbedding's parameter
        #       for e.g. random init;
        #       at the moment, you're supposed
        #       to do actual init on your own
        #       in the client code.
        with torch.no_grad():
            self.weight.fill_(0)

    @weak_script_method
    def forward(self, input):
        shape = list(input.shape) + self.weight.shape[1:]
        return self.weight.index_select(0, input.reshape(-1)).reshape(shape)

    # TODO: backward?
