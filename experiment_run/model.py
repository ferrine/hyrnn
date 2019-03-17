import torch
import torch.nn as nn
import geoopt.manifolds.poincare.math as pmath
import geoopt
import hyrnn
import functools


class RNNBase(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        project_dim,
        cell_type="eucl_rnn",
        embedding_type="eucl",
        decision_type="eucl",
        use_distance_as_feature=True,
        device=None,
        num_layers=1,
        num_classes=1,
        c=1.0,
        order=1,
    ):
        super(RNNBase, self).__init__()
        (cell_type, embedding_type, decision_type) = map(
            str.lower, [cell_type, embedding_type, decision_type]
        )
        if embedding_type == "eucl":
            self.embedding = hyrnn.LookupEmbedding(
                vocab_size, embedding_dim, manifold=geoopt.Euclidean()
            )
        elif embedding_type == "hyp":
            self.embedding = hyrnn.LookupEmbedding(
                vocab_size,
                embedding_dim,
                manifold=geoopt.PoincareBall(c=c).set_default_order(order),
            )
        else:
            raise NotImplementedError(
                "Unsuported embedding type: {0}".format(embedding_type)
            )
        self.embedding_type = embedding_type
        if decision_type == "eucl":
            self.projector = nn.Linear(hidden_dim * 2, project_dim)
            self.logits = nn.Linear(project_dim, num_classes)
        elif decision_type == "hyp":
            self.projector_source = hyrnn.MobiusLinear(
                hidden_dim, project_dim, c=c, order=order
            )
            self.projector_target = hyrnn.MobiusLinear(
                hidden_dim, project_dim, c=c, order=order
            )
            self.logits = hyrnn.MobiusDist2Hyperplane(project_dim, num_classes)
        else:
            raise NotImplementedError(
                "Unsuported decision type: {0}".format(decision_type)
            )
        self.ball = geoopt.PoincareBall(c).set_default_order(order)
        if use_distance_as_feature:
            if decision_type == "eucl":
                self.dist_bias = nn.Parameter(torch.zeros(project_dim))
            else:
                self.dist_bias = geoopt.ManifoldParameter(
                    torch.zeros(project_dim), manifold=self.ball
                )
        else:
            self.register_buffer("dist_bias", None)
        self.decision_type = decision_type
        self.use_distance_as_feature = use_distance_as_feature
        self.softmax = nn.Softmax()
        self.device = device  # declaring device here due to fact we are using catalyst
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.c = c

        if cell_type == "eucl_rnn":
            self.cell = nn.RNN
        elif cell_type == "eucl_gru":
            self.cell = nn.GRU
        elif cell_type == "hyp_gru":
            self.cell = functools.partial(hyrnn.MobiusGRU, c=c)
        else:
            raise NotImplementedError("Unsuported cell type: {0}".format(cell_type))
        self.cell_type = cell_type

        self.cell_source = self.cell(embedding_dim, self.hidden_dim, self.num_layers)
        self.cell_target = self.cell(embedding_dim, self.hidden_dim, self.num_layers)

    def forward(self, input):
        source_input = input[0]
        target_input = input[1]
        alignment = input[2]
        batch_size = alignment.shape[0]

        source_input = self.embedding(source_input)
        target_input = self.embedding(target_input)

        zero_hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device or source_input.device,
        )

        if self.embedding_type == "eucl" and "hyp" in self.cell_type:
            source_input = pmath.expmap0(source_input, c=self.c)
            target_input = pmath.expmap0(target_input, c=self.c)
        elif self.embedding_type == "hyp" and "eucl" in self.cell_type:
            source_input = pmath.logmap0(source_input, c=self.c)
            target_input = pmath.logmap0(target_input, c=self.c)
        # ht: (num_layers * num_directions, batch, hidden_size)
        _, source_hidden = self.cell_source(source_input, zero_hidden)
        _, target_hidden = self.cell_target(target_input, zero_hidden)

        # take hiddens from the last layer
        source_hidden = source_hidden[-1]
        target_hidden = target_hidden[-1][alignment]
        if self.decision_type == "hyp":
            if "eucl" in self.cell_type:
                source_hidden = pmath.expmap0(source_hidden, c=self.c)
                target_hidden = pmath.expmap0(target_hidden, c=self.c)
            source_projected = self.projector_source(source_hidden)
            target_projected = self.projector_target(target_hidden)
            projected = pmath.mobius_add(
                source_projected, target_projected, c=self.ball.c
            )
            if self.use_distance_as_feature:
                dist = (
                    pmath.dist(source_hidden, target_hidden, dim=-1, keepdim=True) ** 2
                )
                bias = pmath.mobius_pointwise_mul(dist, self.dist_bias, c=self.ball.c)
                projected = pmath.mobius_add(projected, bias, c=self.ball.c)
        else:
            if "hyp" in self.cell_type:
                source_hidden = pmath.logmap0(source_hidden, c=self.c)
                target_hidden = pmath.logmap0(target_hidden, c=self.c)
            projected = self.projector(
                torch.cat((source_hidden, target_hidden), dim=-1)
            )
            if self.use_distance_as_feature:
                dist = torch.sum(
                    (source_hidden - target_hidden).pow(2), dim=1, keepdim=True
                )
                bias = self.dist_bias * dist
                projected = projected + bias

        logits = self.logits(projected)
        # CrossEntropy accepts logits
        return logits
