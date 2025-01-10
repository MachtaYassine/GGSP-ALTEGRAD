from NGG.denoiser.denoise_model import DenoiseNN
import torch


class Denoise_GC(DenoiseNN):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super(Denoise_GC, self).__init__(input_dim, hidden_dim, n_layers, n_cond, d_cond)

    def forward(self, x, t, cond):
        cond = self.cond_mlp(cond)
        x = torch.cat((x, cond), dim=-1)
        for i in range(self.n_layers):
            x = self.mlp[i](x)
            if i < self.n_layers-1:
                x = self.bn[i](x)
                x = self.relu(x)
        x = self.tanh(x)
        return x