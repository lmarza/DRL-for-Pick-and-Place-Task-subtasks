import gym
import random
import numpy as np
import argparse
from arguments import get_args
from actorcritic import Actor, second, act, actor
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

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



def train(rank, args, shared_model, counter, lock, optimizer=None):
    
    args2 = get_args()
    # load the model param
    model_path_approach = args2.save_dir + args2.env_name + '/approach.pt'
    o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, model_approach = torch.load(model_path_approach, map_location=lambda storage, loc: storage)
    model_path_manipulate = args2.save_dir + args2.env_name + '/manipulate.pt'
    o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
    model_path_retract = args2.save_dir + args2.env_name + '/retract.pt'
    o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, model_retract = torch.load(model_path_retract, map_location=lambda storage, loc: storage)

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    observation = env.reset()

    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    hlc = Actor()
    # create the actor network
    actor_network_approach = actor(env_params)
    actor_network_approach.load_state_dict(model_approach)
    actor_network_approach.eval()
    actor_network_manipulate = actor(env_params)
    actor_network_manipulate.load_state_dict(model_manipulate)
    actor_network_manipulate.eval()
    actor_network_retract = actor(env_params)
    actor_network_retract.load_state_dict(model_retract)
    actor_network_retract.eval()

    if args.use_cuda:
        hlc.cuda()


   
    for p in hlc.fc1.parameters():
        p.requires_grad = False
    for p in hlc.fc2.parameters():
        p.requires_grad = False
        
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    hlc.train()
    
    done = True       
    for num_iter in count():
        with lock:
            counter.value += 1
        #print(num_iter, counter.value)
        observation = env.reset()
        
        goal = observation['desired_goal']
        objectPos = observation['observation'][3:6]
        object_rel_pos = observation['observation'][6:9]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
        timeStep = 0 #count the total number of timesteps
        grip_pos = -object_rel_pos + objectPos
        
        object_pos_goal = objectPos.copy()
        if grip_pos[0] > objectPos[0]:
            object_pos_goal[0] += 0.003
        else:
            object_pos_goal[0] -= 0.003

        if grip_pos[1] > objectPos[1]:
            object_pos_goal[1] += 0.002
        else:
            object_pos_goal[1] -= 0.002

        object_pos_goal[2] -= -0.031

        if rank == 0:

            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path1)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path1)
        
        hlc.load_state_dict(shared_model.state_dict())
        values, log_probs, rewards, entropies = [], [], [], []
        if done:
            cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        state_inp = torch.from_numpy(env2.observation(observation)).type(FloatTensor)
        #criterion = nn.MSELoss()
        value, y, (hx, cx) = hlc(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        #action_out = torch.tensor([[0]])
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #print(action_out)
        obs = observation["observation"]
        observation_new = observation

        while np.linalg.norm(grip_pos - object_pos_goal) >= 0.031 and timeStep <= 20:
            actions = [0, 0, 0, 0]
            if action_out == 0:
                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                    pi = actor_network_approach(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

            elif action_out == 1:

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                    pi = actor_network_manipulate(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
            
            else: 

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                    pi = actor_network_retract(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

            actions[3] = 0.05
            observation_new, _, _, info = env.step(actions)
            obs = observation_new["observation"]
            g = observation_new["desired_goal"]

            objectPos_new = observation_new["observation"][3:6]
            object_rel_pos_new = observation_new["observation"][6:9]
            objectPos = objectPos_new
            grip_pos_new = -object_rel_pos_new.copy() + objectPos_new.copy()

            grip_pos = grip_pos_new
            object_oriented_goal = object_rel_pos_new
                
            timeStep += 1
            state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
            if timeStep >= 21: 
                break
        
       
        reward = torch.Tensor([-1.0]).type(FloatTensor)
        rewards.append(reward)
        
        value, y, (hx, cx) = hlc(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #action_out = torch.tensor([[1]])

     
        while np.linalg.norm(grip_pos - objectPos) >= 0.015 and timeStep < env._max_episode_steps:
            actions = [0, 0, 0, 0]
            if action_out == 0:
                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                    pi = actor_network_approach(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

            elif action_out == 1:

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                    pi = actor_network_manipulate(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
            
            else: 

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                    pi = actor_network_retract(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
            
            actions[3] = -0.01
            
            observation_new, _, _, info = env.step(actions)
            obs = observation_new["observation"]
            objectPos = observation_new["observation"][3:6]
            object_rel_pos = observation_new["observation"][6:9]
        
            grip_pos_new = -object_rel_pos + objectPos
            grip_pos = grip_pos_new

            timeStep += 1
            state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
            if timeStep >= env._max_episode_steps: 
                break
        
       
        reward = torch.Tensor([-1.0]).type(FloatTensor)
        rewards.append(reward)

        value, y, (hx, cx) = hlc(state_inp, hx, cx)
        prob = F.softmax(y)
        log_prob = F.log_softmax(y, dim=-1)
        act_model = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(act_model))
        action_out = act_model.to(torch.device("cpu"))
        entropies.append(entropy), log_probs.append(log_prob), values.append(value)
        #action_out = torch.tensor([[2]])

        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep < env._max_episode_steps:
            actions = [0, 0, 0, 0]
            if action_out == 0:
                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                    pi = actor_network_approach(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

            elif action_out == 1:

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                    pi = actor_network_manipulate(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
            
            else: 

                with torch.no_grad():
                    #input_tensor = _preproc_inputs(obs, objectPos)
                    input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                    pi = actor_network_retract(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
            actions[3] = -0.01

            # put actions into the environment
            observation_new, _, _, info = env.step(actions)
            obs = observation_new['observation']
            #inputs = process_inputs(obs, g, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args)
            timeStep += 1
            state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
            objectPos = observation_new['observation'][3:6]
            object_rel_pos = observation_new['observation'][6:9]
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
            reward = torch.Tensor([10.0]).type(FloatTensor)
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

            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae).type(FloatTensor)

        total_loss = policy_loss + args.value_loss_coef * value_loss
        optimizer.zero_grad()

        (total_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(hlc.parameters(), args.max_grad_norm)

        ensure_shared_grads(hlc, shared_model)
        optimizer.step()

def test(rank, args, shared_model, counter):
    
    args2 = get_args()
    # load the model param
    model_path_approach = args2.save_dir + args2.env_name + '/approach.pt'
    o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, model_approach = torch.load(model_path_approach, map_location=lambda storage, loc: storage)
    model_path_manipulate = args2.save_dir + args2.env_name + '/manipulate.pt'
    o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, model_manipulate = torch.load(model_path_manipulate, map_location=lambda storage, loc: storage)
    model_path_retract = args2.save_dir + args2.env_name + '/retract.pt'
    o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, model_retract = torch.load(model_path_retract, map_location=lambda storage, loc: storage)

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    observation = env.reset()

    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    hlc = Actor()
    # create the actor network
    actor_network_approach = actor(env_params)
    actor_network_approach.load_state_dict(model_approach)
    actor_network_approach.eval()
    actor_network_manipulate = actor(env_params)
    actor_network_manipulate.load_state_dict(model_manipulate)
    actor_network_manipulate.eval()
    actor_network_retract = actor(env_params)
    actor_network_retract.load_state_dict(model_retract)
    actor_network_retract.eval()
    if args.use_cuda:
        hlc.cuda()
     
    done = True       

    savefile = os.getcwd() + '/train/mario_curves.csv'
    title = ['No. episodes', 'No. of success']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)   

    hlc.eval()
    while True:
        hlc.load_state_dict(shared_model.state_dict())
        hlc.eval()
        ep_num = 0
        success = 0
        num_ep = counter.value
        while ep_num < 50:
            ep_num +=1
            observation = env.reset()            
            #lastObs = observation
            goal = observation['desired_goal']
            objectPos = observation['observation'][3:6]
            object_rel_pos = observation['observation'][6:9]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
            timeStep = 0
            grip_pos = -object_rel_pos + objectPos
        
            object_pos_goal = objectPos.copy()
            if grip_pos[0] > objectPos[0]:
                object_pos_goal[0] += 0.003
            else:
                object_pos_goal[0] -= 0.003

            if grip_pos[1] > objectPos[1]:
                object_pos_goal[1] += 0.002
            else:
                object_pos_goal[1] -= 0.002

            object_pos_goal[2] -= -0.031

            if done:
                cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            else:
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)

            state_inp = torch.from_numpy(env2.observation(observation)).type(FloatTensor)
            value, y, (hx, cx) = hlc(state_inp, hx, cx)
            prob = F.softmax(y)
            act_model = prob.max(-1, keepdim=True)[1].data
            action_out = act_model.to(torch.device("cpu"))


            #print('action_out before approach:', action_out)
            obs = observation["observation"]
            while np.linalg.norm(grip_pos - object_pos_goal) >= 0.031 and timeStep <= 20:
                #env.render()
                actions = [0, 0, 0, 0]
                if action_out == 0:
                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                        pi = actor_network_approach(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()

                elif action_out == 1:

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                        pi = actor_network_manipulate(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                
                else: 

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                        pi = actor_network_retract(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                    

                actions[3] = 0.05

                observation_new, _, _, info = env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]

                objectPos_new = observation_new["observation"][3:6]
                object_rel_pos_new = observation_new["observation"][6:9]
                objectPos = objectPos_new
                grip_pos_new = -object_rel_pos_new + objectPos_new

                grip_pos = grip_pos_new
                object_oriented_goal = object_rel_pos_new
            
                #print('timestep: {},reward eval: {}'.format(timeStep, reward))
                timeStep += 1
                state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
                
            

            value, y, (hx, cx) = hlc(state_inp, hx, cx)
            prob = F.softmax(y)
            act_model = prob.max(-1, keepdim=True)[1].data
            action_out = act_model.to(torch.device("cpu"))
           
            while np.linalg.norm(grip_pos - objectPos) >= 0.015 and timeStep < env._max_episode_steps:
                #env.render()
                actions = [0, 0, 0, 0]
                if action_out == 0:
                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                        pi = actor_network_approach(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()

                elif action_out == 1:

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                        pi = actor_network_manipulate(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                
                else: 

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                        pi = actor_network_retract(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                
                    
                actions[3] = -0.01
                
                observation_new, _, _, info = env.step(actions)
                obs = observation_new["observation"]
                objectPos = observation_new["observation"][3:6]
                object_rel_pos = observation_new["observation"][6:9]
            
                grip_pos_new = -object_rel_pos + objectPos
                grip_pos = grip_pos_new

                timeStep += 1
                state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
                if timeStep >= env._max_episode_steps: 
                    break
        
            value, y, (hx, cx) = hlc(state_inp, hx, cx)
            prob = F.softmax(y)            
            act_model = prob.max(-1, keepdim=True)[1].data            
            action_out = act_model.to(torch.device("cpu"))
           
           
            while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep < env._max_episode_steps:
                #env.render()
                actions = [0, 0, 0, 0]
                if action_out == 0:
                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, object_pos_goal, o_mean_approach, o_std_approach, g_mean_approach, g_std_approach, args2)
                        pi = actor_network_approach(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()

                elif action_out == 1:

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, objectPos, o_mean_manipulate, o_std_manipulate, g_mean_manipulate, g_std_manipulate, args2)
                        pi = actor_network_manipulate(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                
                else: 

                    with torch.no_grad():
                        #input_tensor = _preproc_inputs(obs, objectPos)
                        input_tensor = process_inputs(obs, goal, o_mean_retract, o_std_retract, g_mean_retract, g_std_retract, args2)
                        pi = actor_network_retract(input_tensor)
                        # convert the actions
                        actions = pi.detach().cpu().numpy().squeeze()
                
                actions[3] = -0.01

                # put actions into the environment
                observation_new, _, _, info = env.step(actions)
                obs = observation_new['observation']
                    
                timeStep += 1
                state_inp = torch.from_numpy(env2.observation(observation_new)).type(FloatTensor)
                objectPos = observation_new['observation'][3:6]
                object_rel_pos = observation_new['observation'][6:9]
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
            
            if ep_num % 49==0:            
                print("num episodes {}, success {}".format(num_ep, success*2))
                data = [counter.value, success*2]
                with open(savefile, 'a', newline='') as sfile:
                    writer = csv.writer(sfile)
                    writer.writerows([data])
                    #time.sleep(15)
