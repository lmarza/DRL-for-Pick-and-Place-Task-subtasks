import gym
import numpy as np
import argparse
from actorcritic import Actor, second, act
import torch
import torch.cuda
from torch.autograd import Variable
import torch.multiprocessing as _mp
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
import os
import random
from shared_adam import SharedAdam
import torch.nn as nn
from dense_rewardDDPG_HER import train, test

SAVEPATH1 = os.getcwd() + '/train/actor_params.pth'
SAVEPATH2 = os.getcwd() + '/weights/actor_params.pth'

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument('--save-path1',default=SAVEPATH1,
                    help='model save interval (default: {})'.format(SAVEPATH1))
parser.add_argument('--save-path2',default=SAVEPATH2,
                    help='model save interval (default: {})'.format(SAVEPATH2))
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--max-eps', type=float, default=10000,
                    help='max number of episodes (default: 10000)')
parser.add_argument('--max-steps', type=float, default=50,
                    help='max number of steps per episode (default: 50)')
parser.add_argument('--num-processes', type=int, default=2,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-interval', type=int, default=50,
                    help='model save interval (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
args = parser.parse_args() 

mp = _mp.get_context('spawn')
print("Cuda: " + str(torch.cuda.is_available()))

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    
    args = parser.parse_args()
    env = gym.make("FetchPickAndPlace-v1")
    shared_model = Actor()
    if args.use_cuda:
        shared_model.cuda()
    torch.cuda.manual_seed_all(30)
    
    shared_model.share_memory()

    if os.path.isfile(args.save_path1):
        print('Loading A3C parametets ...')
        pretrained_dict = torch.load(args.save_path1)
        model_dict = shared_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        shared_model.load_state_dict(model_dict)

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    for p in shared_model.fc1.parameters():
        p.requires_grad = False
    for p in shared_model.fc2.parameters():
        p.requires_grad = False

    print ("No of available cores : {}".format(mp.cpu_count())) 

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    print (counter)
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))


    p.start()
    processes.append(p)

    num_procs = args.num_processes
    
    if args.num_processes > 1:
        num_procs = args.num_processes - 1 

    for rank in range(0, num_procs):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()