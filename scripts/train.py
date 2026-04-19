import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from src.utils.tensor_utils import gumbel_log_survival
from src.utils.obj_funct import objFunct

def epoch_train(data, cores, io_pairs, dims, graphs, model, optimizer, sched, num_samples, best_cost, best_mapping, best_order, device):
    loss = 0
    data = data.to(device).permute(1,0,2)
    model.decoding_type = 'sampling-w/o-replacement'
    temp = torch.zeros(data.size(1), data.size(0)).to(device)
    mask = temp==0
    mask=mask.to(device)
    # print(len(io_pairs))
    mappings, log_probs, g_log_probs = model(input=data, num_io=len(io_pairs), mask=mask, num_samples=num_samples+1)
    mappings = mappings[:-1, :].to(device)
    # print(f"Mappings: {mappings}")
    threshold = g_log_probs[-1].to(device)
    log_probs = log_probs[:-1].to(device)
    g_log_probs = g_log_probs[:-1].to(device)
    penalty, orders = objFunct(mappings, data[:, 0, :], cores, io_pairs, dims, graphs)
    penalty = penalty.to(device)
    # take the value of the last sample for threshold
    log_q = gumbel_log_survival(threshold - g_log_probs).detach().to(device)
    log_importance = (log_probs - log_q).detach().to(device)
    wi_s = ((log_importance.unsqueeze(1).repeat(1,log_importance.size(0)) - log_importance.unsqueeze(0)).exp().sum(dim=0) - 1 + torch.exp(log_q)).to(device)
    importance_normalized = F.softmax(log_importance, dim=0).to(device)
    b_s = torch.sum( importance_normalized * penalty).to(device)
    min_penalty = torch.argmax(penalty)
    if penalty[min_penalty] > best_cost:
        best_cost = penalty[min_penalty]
        best_mapping = mappings[min_penalty]
        best_order = orders[min_penalty]
    # print("Penalty:",penalty)
    # print("Baseline:",b_s)
    # print("pen - b_s: ",penalty-b_s)
    # print("log_importance: ", log_importance)
    # print("Weights: ",wi_s)
    # print("Log Probs: ",log_probs)
    loss = torch.sum((1/wi_s) * log_probs * (penalty - b_s)).to(device)
    # loss = (loss+1e5).to(device)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
    optimizer.step()
    sched.step(loss)

    return penalty, min_penalty, loss.item(), best_cost, best_mapping, best_order

def model_train(numcore, model, optimizer, data, cores, io_pairs, dims, graphs, num_samples, batch_size, lr, wt_decay, factor, patience, dropout, n_layers, input_dim, hidden_dim, MAX_EPOCH, device):
    pens = []
    losses = []
    # print("i am in model_train")
    best_cost = float('-inf')
    best_mapping = torch.from_numpy(np.zeros(len(cores))).to(device)
    best_order = list(range(len(cores)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    # samples = generateSamples(num_samples)
   # samples = generateSamples(num_samples)

    for epoch in tqdm(range(MAX_EPOCH)):
        print('Epoch {}: '.format(epoch+1))

        # model.train(True)
        penalty, min_penalty, loss, best_cost, best_mapping, best_order = epoch_train(data, cores, io_pairs, dims, graphs, model, optimizer, scheduler, num_samples, best_cost, best_mapping, best_order)
        
        # model.eval()
        # avg_val_loss, curr_pred = epoch_val()

        pens.append(penalty[min_penalty].item())
        losses.append(loss)
        print('Train loss = {}, Train Penalty = {}, Best Cost = {}, Best mapping = {}, Order = {}'.format(loss, penalty[min_penalty].item(), best_cost, best_mapping, best_order))

        if penalty[min_penalty] > best_cost:
            count_not_decrease += 1
        else:
            count_not_decrease = 0
        if count_not_decrease >= 1000:
            print('Early stopping at epoch {}'.format(epoch))
            break
    epochs = range(1, len(pens) + 1)

# Plot the losses against the indices
    np.save(f"pens_{numcore}cores.npy", pens)
    #plt.plot(epochs, pens, label='Penalty')
    #plt.xlabel('Epochs')
    #plt.ylabel('Penalty')
    #plt.title('Penalty Curve')
    #plt.legend()
    #plt.savefig(f'penalty_curve{idx}.png')
    #prints(pens)

    np.save(f'loss_{numcore}cores.npy', losses)
    #plt.plot(epochs, losses, label='Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Loss Curve')
    #plt.legend()
    #plt.savefig(f'loss_curve{idx}.png')
    #print(losses)
    
    return pens, losses, best_cost, best_mapping