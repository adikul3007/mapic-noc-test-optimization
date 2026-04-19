import torch
from src.utils.obj_funct import objFunct

def test(model, data, cores, io_pairs, dims, graphs, device):
    #model.load_state_dict(torch.load('active_search_rl_32cores.pt'))
    #optimizer.load_state_dict(torch.load('opt_32cores.pt'))
    model.eval()
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=1)
    penalty = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs).to(device)
    return torch.max(penalty).item(), mappings
