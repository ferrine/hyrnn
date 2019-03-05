import torch.jit
import torch.nn
import torch.nn.functional
import geoopt.manifolds.poincare.math as pmath
import geoopt


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
    b_r, b_h, b_z = bias.chunk(3)
    W_hr, W_hh, W_hz = weight_hh.chunk(3)
    Wr_ht = pmath.mobius_matvec(W_hr, hx, c=c)
    Wr_it_b = pmath.mobius_add(pmath.mobius_matvec(W_ir, input, c=c), b_r, c=c)
    r_t = pmath.logmap0(pmath.mobius_add(Wr_ht, Wr_it_b, c=c), c=c).sigmoid()

    Wh_r_hx = pmath.mobius_matvec(W_hh * r_t, hx, c=c)
    Wh_it_b = pmath.mobius_add(pmath.mobius_matvec(W_ih, input, c=c), b_h, c=c)
    h_tilde = pmath.mobius_add(Wh_r_hx, Wh_it_b, c=c)

    Wz_ht = pmath.mobius_matvec(W_hz, hx, c=c)
    Wz_it_b = pmath.mobius_add(pmath.mobius_matvec(W_iz, input, c=c), b_z, c=c)
    z_t = pmath.logmap0(pmath.mobius_add(Wz_ht, Wz_it_b, c=c), c=c).sigmoid()

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c=c)
    delta_h = pmath.mobius_add(-hx, h_tilde, c=c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c=c), c=c)
    return h_out


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
                    self.bias.set_(
                        pmath.expmap0(
                            self.bias.normal_() / 2, c=c
                        )
                    )
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
        self.scale = torch.nn.Parameter(torch.ones(out_features))
        point = torch.randn(out_features, in_features) / 2
        point = pmath.expmap0(point, c=c)
        tangent = torch.randn(out_features, in_features)
        tangent /= tangent.norm(dim=-1, p=2, keepdim=True)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
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


class MobiusGRUCell(torch.nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlin=None, c=1.0):
        super().__init__(input_size, hidden_size, bias=False, num_chunks=3)
        self.ball = geoopt.PoincareBall(c=c)
        if bias:
            self.bias_ih = geoopt.ManifoldParameter(torch.randn(3 * hidden_size)*1e-5, manifold=self.ball)
        self.nonlin = nonlin

    def forward(self, input: torch.Tensor, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        return mobius_gru_cell(
            input=input,
            hx=hx,
            weight_ih=self.weight_ih,
            weight_hh=self.weight_hh,
            bias=self.weight_ih,
            nonlin=self.nonlin,
            c=self.ball.c
        )
