import argparse
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from scripts.data import prep_data
from src.utils.tensor_utils import gumbel_log_survival
from src.utils.obj_funct import objFunct
from src.models.pointer_net import PointerNet
from scripts.train import model_train
from scripts.rl import model_rl
from scripts.inference import test

def main(args):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    mode = args.mode
    
    #---TRAIN---
    if mode is 'train':        
        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        model.load_state_dict(torch.load(f'ordered_active_search_48cores.pt'))
        optimizer.load_state_dict(torch.load(f'ordered_opt_48cores.pt'))
        numcore=[56]  # numcore = [7,10,14,28,32]
        for c in numcore:
            data, cores, io_pairs, dims, graphs = prep_data(c, False)
            pens_list, loss_list, best_cost, best_mapping = model_train(c, model, optimizer, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            torch.save(model.state_dict(), f'ordered_active_search_{c}cores.pt')
            torch.save(optimizer.state_dict(), f'ordered_opt_{c}cores.pt')

    #---RL---
    elif mode is 'rl':
        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        model.load_state_dict(torch.load(args.model_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        numcore = [args.num_cores]
        for c in numcore:
            data, cores, io_pairs, dims, graphs = prep_data(c, False)
            # print(f"io = {len(io_pairs)}")
            model_rl(c, model, optimizer, args.alpha, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            torch.save(model.state_dict(), f'ordered_active_search_rl_{c}cores_{args.alpha}.pt')
            torch.save(optimizer.state_dict(), f'ordered_opt_rl_{c}cores_{args.alpha}.pt')

    #---TEST---
    elif mode is 'test':
        numcore = [args.num_cores]
        model = PointerNet(args.input_dim, args.hidden_dim, args.n_layers, args.dropout, device)
        model.load_state_dict(torch.load(args.model_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
        optimizer.load_state_dict(torch.load(f'fixed_opt_rl_{args.num_cores}cores_0.99.pt'))
        data, cores, io_pairs, dims, graphs = prep_data(args.num_cores, True)
        costs = []
        mappings = []
        for _ in tqdm(range(args.num_test_iter)):
            cost, mapp = test(args.num_cores, model, optimizer, args.alpha, data, cores, io_pairs, dims, graphs, args.num_samples, args.batch_size, args.lr, args.wt_decay, args.factor, args.patience, args.dropout, args.n_layers, args.input_dim, args.hidden_dim, args.epoch)
            costs.append((-1)*cost)
            mapp = mapp[0]
            mappings.append([x.item() for x in mapp])

        print(mappings)
    
    else:
        print("Wrong mode argument")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train/rl/test", type=str, default='test')
    parser.add_argument("--model_path", help="path to model", type=str, default='ordered_active_search_48cores.pt')
    parser.add_argument("--num_samples", help="number of samples", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-8)
    parser.add_argument("--wt_decay", help="decay rate for regularization", type=float, default=1e-5) # og value = 1e-5
    parser.add_argument("--factor", help="factor", type=float, default=0.001) # og value = 0.5
    parser.add_argument("--patience", help="patience", type=float, default=100) # og value = 5
    parser.add_argument("--dropout", help="dropout", type=float, default=0.1) # og value  = 0.1
    parser.add_argument("--n_layers", help="num_layer_encoder", type=int, default=2)
    parser.add_argument("--input_dim", help="input_dim_enc", type=int, default=2)
    parser.add_argument("--hidden_dim", help="hidden_dim", type=int, default=2) # og value =1 
    parser.add_argument("--epoch", help="MAX_EPOCH", type=int, default=10000)
    parser.add_argument("--num_cores", type=int, default=7)
    parser.add_argument("--num_test_iter", help="number of test iterations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.99)
    args = parser.parse_args()
    main(args)
