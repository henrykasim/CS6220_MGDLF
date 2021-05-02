'''Main Function to run MGGCN'''

import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
#from dgl.data import register_data_args
#from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from mggcn import *
#from gcn_mp import GCN
#from gcn_spmv import GCN
from torch.utils.data.dataloader import DataLoader
from functions import *
from torch.autograd import Variable
from utils import progress_bar
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main(args):
    # load and preprocess dataset
    file = 'data/crowd_flow.csv'
    gpu_id = "cuda:" + str(args.gpu)
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset, test_dataset = generate_torch_datasets(file,args.interval)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size)
    checkpointpath = './checkpoint/'
    logpath = './log/'

    ## used for denormalized data
    try:
        feat = np.load('./data/crowd_flow.npy')
    except:
        feat_df = load_dataset(file)
        feat = load_hourly_features(feat_df)

    feat_min = np.min(feat)
    feat_max = np.max(feat)
    print('loading npy feature data',feat_min,feat_max)

    # Load Graph
    graph = load_graph('./data/nyc_adj.csv').to(device)

    # create MGGCN model
    model = MGGCN(graph, 2, args.h1_feats, args.h2_feats, args.h3_feats, args.interval, args.hidden).to(device)

    # define huber loss
    huber_loss = torch.nn.SmoothL1Loss()
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    best_rmse = 0

    # Train Model
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        MSE = 0
        MAE = 0
        total = 0
        training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (args.batch_size, epoch, optimizer.param_groups[0]['lr'])
        #statfile.write('\nTraining Setting: '+training_setting+'\n')
        
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs, targets = torch.transpose(inputs, 0, 1).to(device), targets.to(device)
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            #print(outputs.size(),torch.min(outputs).item(),torch.max(outputs).item())
            #print(targets.size(),torch.min(targets).item(),torch.max(targets).item())
            loss = huber_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
    
            train_loss += loss.item()
            out_denormalized = denormalize(outputs, feat_min, feat_max)
            targets_denormalized = denormalize(targets, feat_min, feat_max)
            #print(torch.max(out_denormalized).item())
            #print(torch.max(targets_denormalized).item())
            MSE += mse_loss(out_denormalized,targets_denormalized).item()*targets.size(0)
            #MSE += np.sqrt(mean_squared_error(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()))
            MAE += mae_loss(out_denormalized,targets_denormalized).item()*targets.size(0)
            #MAE += mean_absolute_error(targets.cpu().numpy(), outputs.cpu().numpy())

            #_, predicted = torch.max(outputs.data, 1)
            #total += targets.size(0)
            #correct += predicted.eq(targets.data).cpu().sum()
            
            progress_bar(batch_idx, len(train_data), 'Loss: %.3f | MSE: %.3f | MAE: %.3f)'
                % (train_loss, np.sqrt(MSE/total), MAE/total))
        RMSE = np.sqrt(MSE/total)
        MAE = MAE/total
        print('Epoch: ',epoch,', - loss:', train_loss,'RMSE: ', RMSE,', MAE: ', MAE)
        #writing training record 
#        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
#                  % (epoch, train_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)  
#        statfile.write(statstr+'\n')   
    
    
    # Test Model
    def test(epoch):
        nonlocal best_rmse
        model.eval()
        test_loss = 0
        MSE = 0
        MAE = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_data):
            inputs, targets = torch.transpose(inputs, 0, 1).to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            #print(inputs.is_cuda)
            outputs = model(inputs)
            loss = huber_loss(outputs, targets)
    
            test_loss += loss.item()
            total += targets.size(0)

            out_denormalized = denormalize(outputs, feat_min, feat_max)
            targets_denormalized = denormalize(targets, feat_min, feat_max)
            #print(torch.max(out_denormalized).item())
            #print(torch.max(targets_denormalized).item())

            MSE += mse_loss(out_denormalized,targets_denormalized).item()*targets.size(0)
            #MSE += np.sqrt(mean_squared_error(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()))
            MAE += mae_loss(out_denormalized,targets_denormalized).item()*targets.size(0)


            progress_bar(batch_idx, len(test_data), ' Loss: %.3f | RMSE: %.3f | MAE RMSE: %.3f)'
                % (test_loss, np.sqrt(MSE/total), MAE/total))
        RMSE = np.sqrt(MSE/total)
        MAE = MAE/total
        print('Epoch: ',epoch,', loss:', test_loss ,'RMSE: ', RMSE,', MAE: ', MAE)
        #statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
          #        % (epoch, test_loss/(batch_idx+1), 100.*(float)(correct)/(float)(total), correct, total, best_acc)
       # statfile.write(statstr+'\n')
        
        # Save checkpoint.
        #acc = 100.*correct/total
        state = {
            'state_dict': model.state_dict(),
            'rmse': RMSE,
            'mae' : MAE,
            'epoch': epoch,           
        }

        #torch.save(state, checkpointpath + modelname + '_last_ckpt.t7')

        #check if current accuarcy is the best
        if RMSE >= best_rmse:  
            print('Saving..')
            save_name = 'mggcn_best_ckpt_hid' + str(args.hidden) + 'itv' + str(args.interval) + '.t7'
            torch.save(state, checkpointpath  + save_name)
            best_mse = RMSE
        
    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    
    #train network
    for epoch in range(args.epochs):
        #statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+')  #open file for writing
        if epoch==30 or epoch==60 or epoch==90:
            decrease_learning_rate()       
        train(epoch)
        if (epoch+1) % 5 == 0:
            test(epoch)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    #register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu_id")
    parser.add_argument("--h1_feats", type=int, default=10,
                        help="h1 feature dimension")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="h1 feature dimension")
    parser.add_argument("--h2_feats", type=int, default=10,
                        help="h2 feature dimension")
    parser.add_argument("--h3_feats", type=int, default=10,
                        help="h3 feature dimension")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--hidden", type=int, default=16,
                        help="number of hidden gru units")
    parser.add_argument("--interval", type=int, default=5,
                        help="number of time steps")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    main(args)
