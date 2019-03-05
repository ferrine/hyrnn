import torch.jit
import torch.nn
import torch.nn.functional
import geoopt


def mobius_linear(input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0):
    if hyperbolic_input:
        output = geoopt.manifolds.poincare.math.mobius_matvec(weight, input, c=c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = geoopt.manifolds.poincare.math.expmap0(output, c=c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = geoopt.manifolds.poincare.math.expmap0(bias, c=c)
        output = geoopt.manifolds.poincare.math.mobius_add(output, bias, c=c)
    if nonlin is not None:
        output = geoopt.manifolds.poincare.math.mobius_fn_apply(nonlin, output, c=c)
    output = geoopt.manifolds.poincare.math.project(output, c=c)
    return output


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, order=1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c).set_default_order(
                    order
                )
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(
                        geoopt.manifolds.poincare.math.expmap0(self.bias.normal_()/2, c=c)
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
        point = torch.randn(out_features, in_features)/2
        point = geoopt.manifolds.poincare.math.expmap0(point, c=c)
        tangent = torch.randn(out_features, in_features)
        tangent /= tangent.norm(dim=-1, p=2, keepdim=True)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere)

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance = geoopt.manifolds.poincare.math.dist2plane(
            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return ("in_features={in_features}, out_features={out_features}, "
                "c={ball.c}, order={ball.default_order}".format(**self.__dict__, ball=self.ball))
