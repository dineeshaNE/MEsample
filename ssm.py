import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    """
    Real State Space Model (causal, recurrent)

    Discrete-time State Space Model:
        h_t = A h_{t-1} + B x_t
        y_t = C h_t + D x_t

✅ True diagonal SSM
✅ Linear time in sequence length
✅ Stable & parallelizable
✅ Exactly how Mamba works internally

complexity O(TD)
    """

    def __init__(self, d_model):
        super().__init__()

        # Diagonal SSM parameters
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Parameter(torch.randn(d_model))
        self.C = nn.Parameter(torch.randn(d_model))
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """

        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t, :]

            # STATE UPDATE (this is the missing part before)
            s = self.A * s + self.B * xt

            # OUTPUT
            yt = self.C * s + self.D * xt
            outputs.append(yt)

        return torch.stack(outputs, dim=1)

"""# quick/sanity test
if __name__ == "__main__":
    x = torch.randn(2, 10, 64)
    model = SimpleSSM(64)

    y = model(x)
    print(y.shape)
"""

class SSM(nn.Module):
    """
    Discrete-time State Space Model:
        h_t = A h_{t-1} + B x_t
        y_t = C h_t
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Learnable state parameters
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)

        outputs = []

        for t in range(T):
            # 🔁 RECURRENCE (this is what makes it an SSM)
            h = self.A * h + self.B(x[:, t])
            y = self.C(h)
            outputs.append(y)

        return torch.stack(outputs, dim=1)