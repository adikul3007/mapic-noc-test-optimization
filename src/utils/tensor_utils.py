import torch
import torch.nn.functional as F

def rearrange(x:torch.Tensor, beams_buf:torch.Tensor)->torch.Tensor:
    batch_size, num_samples = beams_buf.size()
    # print(f"x: {x.shape}")
    if x.dim() == 1:
        return x.view(batch_size, num_samples).gather(1,beams_buf).view(-1)
    elif x.dim() == 2:
        return x.view(batch_size, num_samples, -1)\
            .gather(1,beams_buf.unsqueeze(-1).expand(-1, -1, x.size(-1)))\
                .view(-1, x.size(-1))
    elif x.dim() == 3:
        return x.permute(1,0,2)\
            .view(batch_size, num_samples, x.size(0), x.size(2))\
            .gather(1,beams_buf.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(0), x.size(2)))\
                .view(-1, x.size(0), x.size(2))\
                    .permute(1,0,2)\
                        .contiguous()
    
def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        torch.log(-torch.expm1(-torch.exp(-x)))  # Hope for the best
    )


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    CHECK_VALIDITY = False
    if CHECK_VALIDITY:
        g_inv = _shift_gumbel_maximum(g, Z, dim)
        assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
    g[phi == -float('inf')] = -float('inf')
    return g, argmax


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))

def force_cudnn_initialization(device):
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))