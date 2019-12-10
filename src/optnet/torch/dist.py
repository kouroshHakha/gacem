import torch

def cdf_normal(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
    x_centered = x - mean
    erf_x = (x_centered / torch.tensor(2 ** 0.5) / sigma).erf()
    return 0.5 * (1 + erf_x)

def cdf_logistic(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
    x_centered = x - mean
    sigmoid_x = (x_centered / torch.tensor(2 ** 0.5) / sigma).sigmoid()
    return sigmoid_x

def cdf_uniform(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    eps = torch.tensor(1e-15)
    l = (x - a) / (b - a + eps)
    bot = l.relu()
    cdf = - (torch.tensor(1)-bot).relu() + 1
    return cdf