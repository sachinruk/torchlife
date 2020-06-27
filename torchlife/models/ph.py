# AUTOGENERATED! DO NOT EDIT! File to edit: hazard.PiecewiseHazard.ipynb (unless otherwise specified).

__all__ = ['PieceWiseHazard']

# Cell
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.Tensor.ndim = property(lambda x: x.dim())

# Cell
class PieceWiseHazard(nn.Module):
    """
    Piecewise Hazard where the hazard is constant between breakpoints.
    parameters:
    - breakpoints: time points where hazard would change
    - max_t: maximum point of time to plot to.
    """
    def __init__(self, breakpoints, max_t):
        super().__init__()
        self.logλ = nn.Parameter(torch.randn(len(breakpoints)+1, 1))
        self.register_buffer('breakpoints', torch.Tensor([0] + breakpoints.tolist()))
        bounded_bp = [0] + breakpoints.tolist() + [max_t]
#         self.widths = torch.Tensor(np.diff(bounded_bp).tolist())[:,None]
        self.register_buffer('widths', torch.Tensor(np.diff(bounded_bp).tolist())[:,None])
#         self.zero = torch.zeros(1,1)
        self.prepend_zero = nn.ConstantPad2d((0,0,1,0), 0)
        self.max_t = max_t

    def cumulative_hazard(self, t, t_section):
        """
        Integral of hazard wrt time.
        """
        λ = torch.exp(self.logλ)

        # cumulative hazard
        cum_hazard = λ * self.widths
        cum_hazard = cum_hazard.cumsum(0)
        cum_hazard = self.prepend_zero(cum_hazard)
        cum_hazard_sec = cum_hazard[t_section]

        δ_t = t - self.breakpoints[t_section][:,None]

        return cum_hazard_sec + λ[t_section] * δ_t

    def forward(self, t, t_section, *args):
        return self.logλ[t_section], self.cumulative_hazard(t, t_section)

    def plot_survival_function(self):
        with torch.no_grad():
            # get the times and time sections for survival function
            t_query = np.arange(self.max_t+10)
            breakpoints = self.breakpoints[1:].cpu().numpy()
            t_sec_query = np.searchsorted(breakpoints, t_query)
            # convert to pytorch tensors
            t_query = torch.Tensor(t_query)[:,None]
            t_sec_query = torch.LongTensor(t_sec_query)

            # calculate cumulative hazard according to above
            Λ = self.cumulative_hazard(t_query, t_sec_query)
            surv_fun = torch.exp(-Λ)

        # plot
        plt.figure(figsize=(12,5))
        plt.plot(t_query, surv_fun)
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.show()

    def plot_hazard(self):
        """
        Plot base hazard
        """
        with torch.no_grad():
            width = self.widths.squeeze()
            x = self.breakpoints.squeeze()
            λ = torch.exp(self.logλ)
            y = λ.squeeze()
        # plot
        plt.figure(figsize=(12,5))
        plt.bar(x, y, width, align='edge')
        plt.ylabel('λ')
        plt.xlabel('t')
        plt.show()