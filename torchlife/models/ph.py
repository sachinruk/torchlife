# AUTOGENERATED! DO NOT EDIT! File to edit: 55_hazard.PiecewiseHazard.ipynb (unless otherwise specified).

__all__ = ['PieceWiseHazard']

# Cell
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler

torch.Tensor.ndim = property(lambda x: x.dim())

# Cell
class PieceWiseHazard(nn.Module):
    """
    Piecewise Hazard where the hazard is constant between breakpoints.
    parameters:
    - breakpoints: time points where hazard would change (must include 0 and max possible time)
    """
    def __init__(self, breakpoints:np.array, t_scaler:MaxAbsScaler, **kwargs):
        super().__init__()
        self.t_scaler = t_scaler
        if len(breakpoints.shape) == 1:
            breakpoints = self.t_scaler.transform(breakpoints[:,None])
        else:
            breakpoints = self.t_scaler.transform(breakpoints)
        self.logλ = nn.Parameter(torch.randn(len(breakpoints)-1, 1))
        self.register_buffer('breakpoints', torch.Tensor(breakpoints[:-1]))
        self.register_buffer('widths', torch.Tensor(np.diff(breakpoints, axis=0)))
        self.prepend_zero = nn.ConstantPad2d((0,0,1,0), 0)

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

        δ_t = t - self.breakpoints[t_section]

        return cum_hazard_sec + λ[t_section] * δ_t

    def forward(self, t, t_section, *args):
        return self.logλ[t_section], self.cumulative_hazard(t, t_section)

    def survival_function(self, t:np.array):
        """
        parameters:
        - t: time (do not scale to be between 0 and 1)
        """
        if len(t.shape) == 1:
            t = t[:,None]
        t = self.t_scaler.transform(t)

        with torch.no_grad():
            # get the times and time sections for survival function
            breakpoints = self.breakpoints[1:].cpu().numpy()
            t_sec_query = np.searchsorted(breakpoints.squeeze(), t.squeeze())
            # convert to pytorch tensors
            t_query = torch.Tensor(t)
            t_sec_query = torch.LongTensor(t_sec_query)

            # calculate cumulative hazard according to above
            Λ = self.cumulative_hazard(t_query, t_sec_query)
            return torch.exp(-Λ)

    def hazard(self):
        with torch.no_grad():
            width = self.widths
            breakpoints = self.breakpoints
            λ = torch.exp(self.logλ)
            return (self.t_scaler.inverse_transform(breakpoints).squeeze(),
                    self.t_scaler.inverse_transform(width).squeeze(),
                    λ.squeeze())

    def plot_survival_function(self, t):
        s = self.survival_function(t)
        # plot
        plt.figure(figsize=(12,5))
        plt.plot(t, s)
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.show()

    def plot_hazard(self):
        """
        Plot base hazard
        """
        breakpoints, width, λ = self.hazard()
        # plot
        plt.figure(figsize=(12,5))
        plt.bar(breakpoints, λ, width, align='edge')
        plt.ylabel('λ')
        plt.xlabel('t')
        plt.show()