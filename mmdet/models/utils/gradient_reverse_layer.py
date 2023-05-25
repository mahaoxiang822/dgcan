import torch


class _GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_reverse = _GradientReverseLayer.apply


class GradientReverseLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientReverseLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_reverse(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr