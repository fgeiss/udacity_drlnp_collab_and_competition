import numpy as np
import random
import copy
from collections import namedtuple, deque

import logging
import logging.config
from os import path

from ddpg_extension.model import Actor, Critic

import time
import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_END   = 0.00001
LEARN_EVERY = 20
LEARN_TIMES = 10
PRIO_ALPHA = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent():

    def __init__(self, number_of_agents, state_size, obs_sizes, action_sizes, random_seed):

        self.number_of_agents = number_of_agents
        self.state_size = state_size
        self.obs_sizes = obs_sizes
        self.action_sizes = action_sizes
        self.seed = random.seed(random_seed)
        self.step_counter = 0
        self.epsilon = EPSILON_START
        sum_action_sizes = sum(action_sizes)

        self.agents = [Single_Agent(state_size, sum_action_sizes, obs_size, act_size, random_seed) \
                       for (obs_size,act_size) in zip(obs_sizes,action_sizes)]

        # Shared memory replay buffer
        self.memory = SharedReplayBuffer(number_of_agents, action_sizes, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Set up logging
        self.logger = logging.getLogger('multi_agent')
        fh = logging.FileHandler('multi_agent.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.logger.info('multi_agent object created')

        

    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def act(self,observations, addNoise=True):
        return [agent.act(obs,addNoise) for (agent,obs) in zip(self.agents,observations)]
        
    def step_add_to_memory(self, states, actions, rewards, next_states, dones):
        """Save experience in shared (!) replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(len(states)):
            self.memory.add(states, actions, rewards, next_states, dones)
        self.step_counter += 1
        
        
    def step_learn(self):
        if self.step_counter % LEARN_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(LEARN_TIMES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
    
    def learn(self,experiences, gamma):
        
        indexes, states, actions, rewards, next_states, dones = experiences
        actions_next = [agent.actor_target(next_obs) for (agent, next_obs) in zip(self.agents,next_states)]
        
        actions_next_con = torch.cat(actions_next, dim=1)
        next_states_con = torch.cat(next_states,dim=1)
        actions_con = torch.cat(actions,dim =1)
        states_con = torch.cat(states,dim=1)
        dones_con = torch.cat(dones,dim=1)
        
        # ---------------------------- update critic ---------------------------- #
        # TODO: Understand and clarify retain graph last retain graph also to be eliminated
        for i,agent in enumerate(self.agents):
            Q_targets_next = agent.critic_target(next_states_con, actions_next_con)
            Q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))
            Q_expected = agent.critic_local(states_con, actions_con)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        for i,agent in enumerate(self.agents):
            #actions_pred = agent.actor_local(states[i])
            #actions_pred_con = torch.cat(actions[0:i]+[actions_pred]+actions[i+1:len(actions)],dim=1)
            actions_pred_con = torch.cat([self.agents[j].actor_local(state) if j==i \
                             else self.agents[i].actor_local(state).detach() \
                             for j,state in enumerate(states)], dim=1)
            
            actor_loss = -agent.critic_local(states_con, actions_pred_con).mean()
            # Minimize the loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        for agent in self.agents:
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)
            agent.soft_update(agent.actor_local, agent.actor_target, TAU) 
        
        # ------------------------ update epsilon and noise -------------------- #
        for agent in self.agents:
            agent.update_epsilon()
            agent.noise.reset()

        


class Single_Agent():
    
    def __init__(self, state_size, sum_action_sizes, obs_size, action_size, random_seed):
        
        # initialize the local and target actor and critic network for all agents
        # the actor network takes the observation of the respective agent as input only
        # the critic takes the complete state
        
        self.state_size = state_size
        self.obs_size = obs_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.actor_local = Actor(obs_size, action_size, random_seed).to(device)
        self.actor_target = Actor(obs_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, sum_action_sizes, random_seed).to(device)
        self.critic_target = Critic(state_size, sum_action_sizes, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        # TODO: Add epsilon decay
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = EPSILON_START
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon*EPSILON_DECAY,EPSILON_END)
    
    def hard_update(self, target, source):
        """ Overwrites the parameters from source network in target network """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def act(self, observation, add_noise=True):
        
        """Returns actions for given state as per current policy."""
        observation = torch.from_numpy(observation).float().to(device)
        
        self.actor_local.eval()
        
        scalar = False
        with torch.no_grad():
            if observation.dim() == 1:
                observation.unsqueeze_(0)
                scalar = True
            action = self.actor_local(observation).cpu().data.numpy()
            
            if scalar:
                action =np.squeeze(action)
                
        self.actor_local.train()
        
        if add_noise:
            action += self.epsilon*self.noise.sample()
        
        return np.clip(action, -1, 1)


    def reset(self):
        
        """ Resets the noise function """
        self.noise.reset()
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

        
class SharedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_of_agents, action_sizes, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_of_agents = num_of_agents
        self.action_sizes = action_sizes
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", \
                                     field_names=["index", "observations", "actions", "rewards", "next_observations", "done"])
        self.td_errors = deque(maxlen=buffer_size)
        self.index = 0 #Global counter to help storing and updating td_error values
        self.seed = random.seed(seed)
    
    def add(self, observations, actions, rewards, next_observations, done):
        """Add a new experience to memory."""
        e = self.experience(self.index, observations, actions, rewards, next_observations, done)
        self.memory.append(e)
        self.index+=1

    
    def sample(self):
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        indexes = [e.index for e in experiences if e is not None]
                
        # TODO: Make this part somewhat more elegant
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        for i in range(self.num_of_agents):
            observations.append(\
                torch.from_numpy(np.vstack([e.observations[i] for e in experiences if e is not None])).float().to(device))
            actions.append(\
                torch.from_numpy(np.vstack([e.actions[i] for e in experiences if e is not None])).float().to(device))
            rewards.append(\
                torch.from_numpy(np.vstack([e.rewards[i] for e in experiences if e is not None])).float().to(device))
            next_observations.append(\
                torch.from_numpy(np.vstack([e.next_observations[i] for e in experiences if e is not None])).float().to(device))
            dones.append(\
                torch.from_numpy(np.vstack([e.done[i] for e in experiences \
                                            if e is not None]).astype(np.uint8)).float().to(device))
        

        return (indexes, observations, actions, rewards, next_observations, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

