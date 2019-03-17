import torch.jit
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.poincare.math as pmath
import geoopt
from .util import extract_last_states


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    c=1.0,
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, c=c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, c=c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, c=c)
        output = pmath.mobius_add(output, bias, c=c)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, c=c)
    output = pmath.project(output, c=c)
    return output


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    Wr_ht = pmath.mobius_matvec(W_hr, hx, c=c)
    Wr_it = pmath.mobius_matvec(W_ir, input, c=c)
    Wr_ht_plus_Wr_it = pmath.mobius_add(Wr_ht, Wr_it, c=c)
    r_t = pmath.logmap0(pmath.mobius_add(Wr_ht_plus_Wr_it, b_r, c=c), c=c).sigmoid()

    Wh_r_hx = pmath.mobius_matvec(W_hh * r_t.unsqueeze(1), hx, c=c)
    Wh_it = pmath.mobius_matvec(W_ih, input, c=c)
    Wh_r_hx_plus_Wh_it_b = pmath.mobius_add(Wh_r_hx, Wh_it, c=c)
    h_tilde = pmath.mobius_add(Wh_r_hx_plus_Wh_it_b, b_h, c=c)

    Wz_ht = pmath.mobius_matvec(W_hz, hx, c=c)
    Wz_it = pmath.mobius_matvec(W_iz, input, c=c)
    Wz_ht_plus_Wz_it = pmath.mobius_add(Wz_ht, Wz_it, c=c)
    z_t = pmath.logmap0(pmath.mobius_add(Wz_ht_plus_Wz_it, b_z, c=c), c=c).sigmoid()

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c=c)
    delta_h = pmath.mobius_add(-hx, h_tilde, c=c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c=c), c=c)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, c=c)
    else:
        hx = h0
    if not hyperbolic_input:
        input = pmath.expmap0(input, c=c)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
        outs = torch.stack(outs)
    else:
        for t in range(batch_sizes.size(0)):
            ix, input = input[:batch_sizes[t]], input[batch_sizes[t]:]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx[:batch_sizes[t]],
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
        outs = torch.cat(outs)
    return outs


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
        order=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c).set_default_order(order)
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() / 2, c=c))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=self.ball.c,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
            if self.hyperbolic_bias:
                info += ", order={}".format(self.ball.default_order)
        return info


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0, order=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=c).set_default_order(order)
        self.sphere = sphere = geoopt.manifolds.Sphere().set_default_order(order)
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 2
        point = pmath.expmap0(point, c=c)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball).proj_()
        self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere)

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance = pmath.dist2plane(
            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={ball.c}, order={ball.default_order}".format(
                **self.__dict__, ball=self.ball
            )
        )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlin=None,
        c=1.0,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = torch.nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            bias = torch.randn(3, hidden_size) * 1e-5
            self.bias = geoopt.ManifoldParameter(
                pmath.expmap0(bias, c=self.ball.c), manifold=self.ball
            )
        else:
            self.register_parameter("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input
        else:
            batch_sizes = None
        if h0 is None:
            h0 = input.new_zeros(input.size(1), self.hidden_size, requires_grad=False)
        outs = mobius_gru_loop(
            input=input,
            h0=h0,
            weight_ih=self.weight_ih,
            weight_hh=self.weight_hh,
            bias=self.bias,
            c=self.ball.c,
            hyperbolic_hidden_state0=self.hyperbolic_hidden_state0,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            batch_sizes=batch_sizes
        )
        if is_packed:
            ht = extract_last_states(outs, batch_sizes)
            outs = torch.nn.utils.rnn.PackedSequence(outs, batch_sizes)
            # TODO: separate out into a util function
            # TODO: find means to vectorize, or at least rewrite in C++
        else:
            ht = outs[-1]
        # default api assumes
        # outs: seq_len x batch_size x hidden_size
        # ht  : 1 x batch_size x hidden_size
        return outs, ht.unsqueeze(0)

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)
