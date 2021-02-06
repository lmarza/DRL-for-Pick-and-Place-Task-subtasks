import gym
import random
import numpy as np
import argparse
from actorcritic import Actor, second, act
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
import matplotlib.pyplot as plt
from torch.distributions import Normal
import os
import random
import torch.nn as nn
from itertools import count
import time
import csv

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer=None):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = Actor()
    model2 = second()

    if args.use_cuda:
        model.cuda()
        model2.cuda()

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
        
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    model.train()
    model2.eval()
    done = True       
    for num_iter in count():
        with lock:
            counter.value += 1
        #print(num_iter, counter.value)
        lastObs = env.reset()
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
        timeStep = 0 #count the total number of timesteps
        if rank == 0:

            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path1)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path1)
        
        model.load_state_dict(shared_model.state_dict())
        values, log_probs, rewards, entropies = [], [], [], []
        if done:
            cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        state_inp = torch.from_numpy(env2.observation(lastObs)).type(FloatTensor)
        #criterion = nn.MSELoss()
        value, y, (hx, cx) = model(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        #action_out = torch.tensor([[0]])
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #print(action_out)
        while np.linalg.norm(object_oriented_goal) >= 0.015 and timeStep <= env._max_episode_steps:
            #env.render()
            action = [0, 0, 0, 0]
            act_tensor= act(state_inp, action_out, model2)      
            #print(act_tensor)     
            for i in range(len(object_oriented_goal)):
                action[i] = act_tensor[i].cpu().detach().numpy()

            object_oriented_goal = object_rel_pos.copy()            
            object_oriented_goal[2] += 0.03
            
            action[3] = 0.05
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
            if timeStep >= env._max_episode_steps: 
                reward = torch.Tensor([-1.0]).type(FloatTensor)
                break
        
        if timeStep < env._max_episode_steps: 
            reward = torch.Tensor([1.0]).type(FloatTensor)
        rewards.append(reward)
        
        value, y, (hx, cx) = model(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #action_out = torch.tensor([[1]])
        while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
            #env.render()
            action = [0, 0, 0, 0]
            act_tensor= act(state_inp, action_out, model2)   

            for i in range(len(object_oriented_goal)):
                action[i] = act_tensor[i].cpu().detach().numpy()
            
            action[3]= -0.01 
           
            
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
            if timeStep >= env._max_episode_steps: 
                reward = torch.Tensor([-1.0]).type(FloatTensor)
                break
        
        if timeStep < env._max_episode_steps: 
            reward = torch.Tensor([1.0]).type(FloatTensor)
        rewards.append(reward)

        value, y, (hx, cx) = model(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #action_out = torch.tensor([[2]])
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
            #env.render()
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
            if timeStep >= env._max_episode_steps: 
                break

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
            reward = torch.Tensor([1.0]).type(FloatTensor)
        else:
            reward = torch.Tensor([-1.0]).type(FloatTensor)
        rewards.append(reward)
        
        R = torch.zeros(1, 1)
        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor)

        total_loss = policy_loss + args.value_loss_coef * value_loss
        optimizer.zero_grad()

        (total_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

def test(rank, args, shared_model, counter):
    
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = Actor()
    model2 = second()
    if args.use_cuda:
        model.cuda()
        model2.cuda()

    done = True       
    

    savefile = os.getcwd() + '/train/mario_curves.csv'
    title = ['No. episodes', 'No. of success']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)   

    if os.path.isfile(args.save_path2):
        print('Loading second parametets ...')
        pretrained_dict = torch.load(args.save_path2)
        model_dict2 = model2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict2}
        model_dict2.update(pretrained_dict) 
        model2.load_state_dict(model_dict2)

    model2.eval()
    model.eval()
    while True:
        model.load_state_dict(shared_model.state_dict())
        model.eval()
        ep_num = 0
        success = 0
        num_ep = counter.value
        while ep_num < 50:
            ep_num +=1            
            lastObs = env.reset()
            goal = lastObs['desired_goal']
            objectPos = lastObs['observation'][3:6]
            object_rel_pos = lastObs['observation'][6:9]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
            timeStep = 0
            if done:
                cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            else:
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)

            state_inp = torch.from_numpy(env2.observation(lastObs)).type(FloatTensor)
            value, y, (hx, cx) = model(state_inp, hx, cx)
            prob = F.softmax(y)
            act_model = prob.max(-1, keepdim=True)[1].data
            action_out = act_model.to(torch.device("cpu"))
            #action_out = torch.tensor([[0]])
            while np.linalg.norm(object_oriented_goal) >= 0.015 and timeStep <= env._max_episode_steps:
                #env.render()
                action = [0, 0, 0, 0]
                act_tensor= act(state_inp, action_out, model2)      
                #print(act_tensor)     
                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()

                object_oriented_goal = object_rel_pos.copy()            
                object_oriented_goal[2] += 0.03
                
                action[3] = 0.05
                obsDataNew, reward, done, info = env.step(action)
                timeStep += 1
                objectPos = obsDataNew['observation'][3:6]
                object_rel_pos = obsDataNew['observation'][6:9]
                state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
                if timeStep >= env._max_episode_steps: 
                    break
    
            value, y, (hx, cx) = model(state_inp, hx, cx)
            prob = F.softmax(y)
            act_model = prob.max(-1, keepdim=True)[1].data
            action_out = act_model.to(torch.device("cpu"))
            #action_out = torch.tensor([[1]])
            while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
                #env.render()
                action = [0, 0, 0, 0]
                act_tensor= act(state_inp, action_out, model2)

                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3]= -0.01 
                #if action_out ==0:
                    #action[4] = act_tensor[3].cpu().detach().numpy()
                
                obsDataNew, reward, done, info = env.step(action)
                timeStep += 1

                objectPos = obsDataNew['observation'][3:6]
                object_rel_pos = obsDataNew['observation'][6:9]
                state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
                if timeStep >= env._max_episode_steps: 
                    break
            
            value, y, (hx, cx) = model(state_inp, hx, cx)
            prob = F.softmax(y)            
            act_model = prob.max(-1, keepdim=True)[1].data            
            action_out = act_model.to(torch.device("cpu"))
            #action_out = torch.tensor([[2]])
            while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
                #env.render()
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
                if timeStep >= env._max_episode_steps: 
                    break
            
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
                #lastObs = env.reset()
                if ep_num % 49==0:            
                    print("num episodes {}, success {}".format(num_ep, success*2))
                    data = [counter.value, success*2]
                    with open(savefile, 'a', newline='') as sfile:
                        writer = csv.writer(sfile)
                        writer.writerows([data])
                        #time.sleep()

                