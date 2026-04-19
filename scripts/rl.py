import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from src.utils.tensor_utils import gumbel_log_survival
from src.utils.obj_funct import objFunct

def epoch_rl(data, cores, io_pairs, dims, graphs, model, optimizer, num_samples, best_cost, best_mapping, alpha, device):
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    n = num_samples//data.size(1)
    b = 0
    losses = []
    for i in tqdm(range(n)):
        mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=2)
        mappings = mappings[:-1, :].to(device)
        threshold = g_log_probs[-1].to(device)
        log_probs = log_probs[:-1].to(device)
        g_log_probs = g_log_probs[:-1].to(device)
        penalty, order = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs)
        penalty = penalty.to(device)
        #print("Order  = ", order)
        # print(f"Mappings: {len(mappings)}")
        # print(f"pen: {len(penalty)}")
        log_q = gumbel_log_survival(threshold - g_log_probs).detach().to(device)
        log_importance = (log_probs - log_q).detach().to(device)
        wi_s = ((log_importance.unsqueeze(1).repeat(1,log_importance.size(0)) - log_importance.unsqueeze(0)).exp().sum(dim=0) - 1 + torch.exp(log_q)).to(device)
        importance_normalized = F.softmax(log_importance, dim=0).to(device)
        b_s = torch.sum( importance_normalized * penalty).to(device)
        min_penalty = torch.argmax(penalty)
        if penalty[min_penalty] > best_cost:
            best_cost = penalty[min_penalty]
            best_mapping = mappings[min_penalty]
        loss = torch.sum((1/wi_s) * log_probs * (penalty - b)).to(device)
        print(f"RL Loss = {loss},  Best Cost = {best_cost}, Best mapping = {best_mapping}")
        # print(f"Best Mapping: {best_mapping}")
        # print(f"Best Cost: {best_cost}")
        # print(f"threshold: {threshold}")
        # print(f"imp_norm: {importance_normalized}")
        # print(f"g_log_probs: {g_log_probs}")
        # print(f"log_q: {log_q}")
        # print("Penalty:",penalty)
        # print("Baseline:",b_s)
        # print("pen - b_s: ",penalty-b_s)
        # print("log_importance: ", log_importance)
        # print("Weights: ",wi_s)
        # print("Log Probs: ",log_probs)
        # print(f"Loss: {loss}")
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        # sched.step(loss)
        b = alpha*b + (1-alpha)*b_s.mean(dim=0)
    
    return best_cost, best_mapping, losses

def model_rl(numcore, model, optimizer, alpha, data, cores, io_pairs, dims, graphs, num_samples, batch_size, lr, wt_decay, factor, patience, dropout, n_layers, input_dim, hidden_dim, MAX_EPOCH, device):
    best_cost = float('-inf')
    best_mapping = torch.from_numpy(np.zeros(len(cores))).to(device)
    # alpha = 0.99
    model.load_state_dict(torch.load('ordered_active_search_56cores.pt'))
    optimizer.load_state_dict(torch.load('ordered_opt_56cores.pt'))
    best_cost, best_mapping, losses = epoch_rl(data, cores, io_pairs, dims, graphs, model, optimizer, num_samples, best_cost, best_mapping, alpha)
    losses = np.array(losses)
    np.save(f'loss_rl_{numcore}cores_{alpha}.npy', losses)
    # print(f"Best Mapping: {best_mapping}")
