import torch

import math



def minmax_range_setting(w: torch.Tensor, is_asymmetric: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

    if is_asymmetric:
        max, min = torch.max(w), torch.min(w)
    else:
        max, min = (torch.max(torch.abs(w))), -(torch.max(torch.abs(w)))

    return max, min

def MSE_analysis_range_setting(w: torch.Tensor, n_bit: int, is_asymmetric: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

    Alpha = torch.arange(1, 80, 0.01)
    std = torch.std(w)

    analysis = SymGaussianClippingAnalysis(w, Alpha, std, n_bit)

    winner_alpha = Alpha[torch.argmin(torch.tensor(analysis))] 

    max = (2*winner_alpha)
    min = -(2*winner_alpha) # to do for asymetric

    return torch.tensor(max), torch.tensor(min)

def MSE_simulation_range_setting(w: torch.Tensor, n_bit: int, is_asymmetric: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

    Alpha = torch.arange(1, 40, 0.1)
    std = torch.std(w)

    simulation = SymGaussianClippingSimulation(w, Alpha, std, n_bit)

    winner_alpha = Alpha[torch.argmin(torch.tensor(simulation))] 

    max = (2*winner_alpha)
    min = -(2*winner_alpha) # to do for asymetric

    return torch.tensor(max), torch.tensor(min)

def uniform_quantizer(x, S):
    xQ = torch.round(x / S) * S
    return xQ

# TO DO
def SymGaussianClippingAnalysis(x_float, Alpha, sigma,bitWidth):
    Analysis = []
    for alpha in Alpha:
        clipping_mse = (sigma**2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma*torch.sqrt(torch.tensor(2.0))))) - torch.sqrt(torch.tensor(2.0/torch.pi)) * alpha * sigma*(torch.e ** ((-1)*(0.5* (alpha ** 2))/sigma**2))
        quant_mse = (alpha ** 2) / (3 * (2 ** (2 * bitWidth)))
        mse = clipping_mse + quant_mse
        Analysis.append(mse)
    return Analysis

def SymGaussianClippingSimulation(x_float, Alpha, sigma,bitWidth):
    simulations = []
    for alpha in Alpha:
        s = torch.clone(x_float)
        S = (2*alpha)/(2**bitWidth)

        # clipping
        s[s > alpha] = alpha
        s[s < -alpha] = -alpha

        # quantization
        s = uniform_quantizer(s, S)

        mse = ((s - x_float) ** 2).mean()
        simulations.append(mse)
    return simulations