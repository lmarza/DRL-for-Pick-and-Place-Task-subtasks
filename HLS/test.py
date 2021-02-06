import gym
import random
import numpy as np
import argparse
from actorcritic import Actor, second, act
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import random
import torch.nn as nn


SAVEPATH1 = os.getcwd() + '/train/actor_params.pth'
#SAVEPATH2 = os.getcwd() + '/train/saved_weights.pth'
SAVEPATH2 = os.getcwd() + '/weights/actor_params.pth'

env = gym.make("FetchPickAndPlace-v1")
env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument('--save-path1',default=SAVEPATH1,
                    help='model save interval (default: {})'.format(SAVEPATH1))
parser.add_argument('--save-path2',default=SAVEPATH2,
                    help='model save interval (default: {})'.format(SAVEPATH2))
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
args = parser.parse_args() 

model = Actor()
model2 = second()
if args.use_cuda:
    model.cuda()
    model2.cuda()
torch.cuda.manual_seed_all(21)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if os.path.isfile(args.save_path1):
    print('Loading A3C parametets ...')
    model.load_state_dict(torch.load(args.save_path1))

if os.path.isfile(args.save_path2):
    print('Loading second parametets ...')
    pretrained_dict = torch.load(args.save_path2)
    model_dict2 = model2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict2}
    model_dict2.update(pretrained_dict) 
    model2.load_state_dict(model_dict2)

for p in model.fc1.parameters():
    p.requires_grad = False
for p in model.fc2.parameters():
    p.requires_grad = False

FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

model.eval()
model2.eval()

max_eps = 200000
max_steps = 50
ep_numb = 0
done = True
success = 0         

while ep_numb < max_eps:
    ep_numb +=1
    lastObs = env.reset()
    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][3:6]
    object_rel_pos = lastObs['observation'][6:9]
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
    timeStep = 0 #count the total number of timesteps
    state_inp = torch.from_numpy(env2.observation(lastObs)).type(FloatTensor)
    if done:
        cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
        hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
    else:
        cx = Variable(cx.data).type(FloatTensor)
        hx = Variable(hx.data).type(FloatTensor)
    
    value, y, (hx, cx) = model(state_inp, hx, cx)
    prob = F.softmax(y)
    
    act_model = prob.max(-1, keepdim=True)[1].data
    
    #action_out = act_model.to(torch.device("cpu"))
    action_out = torch.tensor([[1]])
    while np.linalg.norm(object_oriented_goal) >= 0.01 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]
        act_tensor= act(state_inp, action_out, model2)      
        #print(act_tensor)     
        for i in range(3):
            action[i] = act_tensor[i].cpu().detach().numpy()

        object_oriented_goal = object_rel_pos.copy()            
        object_oriented_goal[2] += 0.03
        
        action[3] = 0.05
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        if timeStep >= env._max_episode_steps: break
    
    value, y, (hx, cx) = model(state_inp, hx, cx)
    prob = F.softmax(y)
    act_model = prob.max(-1, keepdim=True)[1].data
    #action_out = act_model.to(torch.device("cpu"))
    action_out = torch.tensor([[0]])
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        act_tensor= act(state_inp, action_out, model2)   

        for i in range(len(object_oriented_goal)):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3]= -0.01 
        #if action_out == 0:
            #action[5] = act_tensor[3].cpu().detach().numpy()
        
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        if timeStep >= env._max_episode_steps: break
    
    value, y, (hx, cx) = model(state_inp, hx, cx)
    prob = F.softmax(y)
    act_model = prob.max(-1, keepdim=True)[1].data
    #action_out = act_model.to(torch.device("cpu"))
    action_out = torch.tensor([[2]])
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
        env.render()
        action = [0, 0, 0, 0]
        act_tensor= act(state_inp, action_out, model2)

        for i in range(len(goal - objectPos)):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3] = -0.01
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        if timeStep >= env._max_episode_steps: break
    
    while True: #limit the number of timesteps in the episode to a fixed duration
        #env.render()
        action = [0, 0, 0, 0]
        action[3] = -0.01 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    if info['is_success'] == 1.0:
        success +=1
    if done:
        if ep_numb % 100==0:            
            print("num episodes {}, success {}".format(ep_numb, success))