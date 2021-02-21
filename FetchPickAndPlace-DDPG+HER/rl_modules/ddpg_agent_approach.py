import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import time

"""
ddpg with HER (MPI-version)

"""


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)

        # create the replay buffer for approach
        self.bufferApproach = replay_buffer(self.env_params, self.args.buffer_size,self.her_module.sample_her_transitions)
    
        # create the normalizer
        self.o_norm = normalizer(size=env_params["obs"], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params["goal"], default_clip_range=self.args.clip_range)

        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network

        """
        print("epoch \t episode \t steps \t success_rate %")
        stable = False
        for epoch in range(self.args.n_epochs):  # 50
            for episode in range(self.args.n_cycles):  # 50
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
               
                for _ in range(self.args.num_rollouts_per_mpi):  # 2
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    while len(ep_obs) < 50:
                        timeStep = 0
                        observation = self.env.reset()
                        obs = observation["observation"]
                        ag = observation["achieved_goal"]
                        goal = observation["desired_goal"]
                        objectPos = observation["observation"][3:6]
                        object_rel_pos = observation["observation"][6:9]
                        object_oriented_goal = object_rel_pos.copy()
                        object_oriented_goal[2] += 0.03 
                      
                        'the goal is equal to the object_rel_pos - position of the object relative to the gripper, object_rel_pos = object_pos - grip_pos '
                        'our goal is the block for approach part'
                        grip_pos = -object_rel_pos + objectPos
                        ag = grip_pos
                    
                    
                        "calcute target point for approach training"
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

                       
                        """
                        train approach part 
                        start to collect samples fo approach

                        """
                        while np.linalg.norm(grip_pos - object_pos_goal) >= 0.031 and timeStep <= 20:
                            #self.env.render()
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, object_pos_goal)
                                pi = self.actor_network(input_tensor)
                                action = self._select_actions(pi)

                            # feed the actions into the environment
                            action[3] = 0.05
                            observation_new, reward, _, info = self.env.step(action,object_pos_goal)
                            
                            'calculate the new gripper position after one step action and update achieved_goal'
                            objectPos_new = observation_new["observation"][3:6]
                            object_rel_pos_new = observation_new["observation"][6:9]
                            grip_pos_new = -object_rel_pos_new + objectPos_new
                            
                            obs_new = observation_new["observation"]
                            ag_new = grip_pos_new 
                        
                            # append rollouts
                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(object_pos_goal.copy())
                            ep_actions.append(action.copy())
                        
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                            grip_pos = grip_pos_new
                            object_oriented_goal = object_rel_pos_new

                            if len(ep_obs) > 49:
                                break

                            timeStep += 1
                        
                        
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)    

                # store the episodes
                self.bufferApproach.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network("approach")
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent()
            print("{} \t {} \t\t {} \t {}".format(epoch, (episode+1)*(epoch+1), 2500*(epoch+1), success_rate),)

            
            if success_rate > 98:
                torch.save([self.o_norm.mean,self.o_norm.std,self.g_norm.mean,self.g_norm.std,self.actor_network.state_dict()],self.model_path + "/approach.pt")
                
            

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += (self.args.noise_eps * self.env_params["action_max"] * np.random.randn(*action.shape))
        action = np.clip(action, -self.env_params["action_max"], self.env_params["action_max"])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params["action_max"],high=self.env_params["action_max"],size=self.env_params["action"])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
       
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions= episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {
            "obs": mb_obs,
            "ag": mb_ag,
            "g": mb_g,
            "actions": mb_actions,
            "obs_next": mb_obs_next,
            "ag_next": mb_ag_next
        }

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions["obs"], transitions["g"]
        # pre process the obs and g
        transitions["obs"], transitions["g"] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions["obs"])
        self.g_norm.update(transitions["g"])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    # preproc obs and goal
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, subtask):
        # sample the episodes
        if subtask == "approach":
            transitions = self.bufferApproach.sample(self.args.batch_size)
        elif subtask == "manipulate":
            transitions = self.bufferManipulate.sample(self.args.batch_size)
        else:
            transitions = self.bufferRetract.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += (self.args.action_l2 * (actions_real / self.env_params["action_max"]).pow(2).mean())

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        tot_success = 0
        for test in range(100):  # 10
            timeStep = 0
            observation = self.env.reset()
            obs = observation["observation"]
            goal = observation["desired_goal"]
            objectPos = observation["observation"][3:6]
            object_rel_pos = observation["observation"][6:9]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object
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

            "Test approach part"
            while np.linalg.norm(grip_pos - object_pos_goal) >= 0.031 and timeStep <= 20:
                #self.env.render()
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, object_pos_goal)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                actions[3] = 0.05

                observation_new, reward, _, info = self.env.step(actions, object_pos_goal)
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
                
                if timeStep >= self.env._max_episode_steps:
                    break
         
            
            '''
            Manipulate hard-engineered solution
            '''
            while np.linalg.norm(grip_pos - objectPos) >= 0.015 and timeStep < self.env._max_episode_steps:
                #self.env.render()
                action = [0, 0, 0, 0]
                for i in range(len(object_rel_pos)):
                    action[i] = object_rel_pos[i] * 6

                action[3] = -0.01

                obsDataNew, reward, done, info = self.env.step(action, objectPos)
                timeStep += 1

                objectPos = obsDataNew["observation"][3:6]
                object_rel_pos = obsDataNew["observation"][6:9]
                objectPos = objectPos_new
                grip_pos_new = -object_rel_pos + objectPos_new

                grip_pos = grip_pos_new
                

                if timeStep >= self.env._max_episode_steps:
                    break
            
          
            '''
            Retract hard-engineered solution
            '''
            while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep < self.env._max_episode_steps:
                #self.env.render()
                action = [0, 0, 0, 0]
                for i in range(len(goal - objectPos)):
                    action[i] = (goal - objectPos)[i] * 6

                action[3] = -0.02
                obsDataNew, reward, done, info = self.env.step(action, goal)
                timeStep += 1

                objectPos = obsDataNew["observation"][3:6]
                object_rel_pos = obsDataNew["observation"][6:9]

                if timeStep >= self.env._max_episode_steps:
                    break
            
           

            '''
            To target point hard-engineered solution
            '''
            while True: 
                #self.env.render()
                action = [0, 0, 0, 0]
                action[3] = -0.01  # keep the gripper closed

                obsDataNew, reward, done, info = self.env.step(action,goal)
                timeStep += 1

                objectPos = obsDataNew["observation"][3:6]
                object_rel_pos = obsDataNew["observation"][6:9]

                # limit the number of timesteps in the episode to a fixed duration
                if timeStep >= self.env._max_episode_steps:
                    break

            if info["is_success"] == 1:
                tot_success += 1

        return tot_success

