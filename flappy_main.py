#from __future__ import print_function
#import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions import Categorical

from ple.games.flappybird import FlappyBird as GameEnv
from ple import PLE
import numpy as np

from agent import *
from preprocessor import *


model_filename = "model.pytorch"
render = True
crop_size = 407
side_size = 32
input_shape = (side_size,side_size)
number_of_frames = 2
stacked_frame_shape = (1, number_of_frames, side_size, side_size)
input_size = input_shape[0] * input_shape[1]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
eps = np.finfo(np.float32).eps.item()
save_every_episodes = 50

class GameLearner():

    def __init__(self, game, agent, render=True):
        self.p = PLE(game, fps=30, display_screen=render)
        self.agent = agent
        self.episode_no  = 0

    def _discount_rewards(self, episode_rewards):
        '''
        running = []
        for i in range(len(episode_rewards)):
            if (episode_rewards[i] != 0):
                episode_rewards[running] = episode_rewards[i]
                running = []
            else:
                running += [ i ]
        '''
        episode_rewards[episode_rewards==0]=0.1
        episode_rewards[episode_rewards==-5]=-1
        #print(episode_rewards)

        R = 0
        rewards = []
        for r in reversed(range(len(episode_rewards))):
            R = r + 0.7 * R
            rewards.insert(0, R)

        episode_rewards = np.array(rewards)
        episode_rewards = (episode_rewards - episode_rewards.mean()) / (episode_rewards.std() + eps)
        return episode_rewards


    def reinforced_learning(self):
        self.p.init()
        self.p.reset_game()
        reward = 0
        #self.agent.max_pipes = 0
        cont_img = 0
        while(True):
            #housekeeping for new episode
            self.episode_no+=1
            passed_pipes = 0
            done = False
            episode_states =    []
            episode_rewards =   []
            episode_log_probs = []
            episode_probs =        []
            episode_actions = []
            stacked_cont = 0
            stacked_frames = []
            random.seed(datetime.now())
            k = random.randint(0, 40) #a different start position for the policy
            frames = 0
            while not(done): #check episode is not done
                stacked_cont+=1
                if self.p.game_over(): #a bad sequence of moves, done
                    done=True
                else:
                    x = self.p.getScreenRGB()
                    x = pre_process(x, input_shape, crop_size)
                    stacked_frames += [ x ]
                    if stacked_cont == number_of_frames:
                        #Preprocessor.save_img(x, str(cont_img) + ".jpg" )
                        cont_img+=1
                        stacked_frames = np.array(stacked_frames).reshape(stacked_frame_shape)
                        #if frames <= k:
                        #    reward = self.p.act(random.choice([self.agent.NONE])) #, self.agent.UP]))
                        #else:
                        prob, log_prob, action = self.agent.pick_action(stacked_frames)
                        action = self.agent.translate_o_to_action(action)
                        reward = self.p.act(action)
                        if reward > 0:
                            passed_pipes+=1
                            if self.agent.max_pipes < passed_pipes:
                                self.agent.max_pipes = passed_pipes
                        episode_states += [stacked_frames]
                        episode_rewards += [reward]
                        episode_log_probs += [log_prob]
                        episode_probs += [prob]
                        episode_actions += [action]
                        frames+=1
                        stacked_cont = 0

            self.p.reset_game()
            if self.episode_no % save_every_episodes == 0:
                self.agent.save(model_filename)
            episode_rewards = self._discount_rewards(np.array(episode_rewards))
            #episode_rewards[episode_rewards == -5] = -1
            total_reward = np.sum(episode_rewards)
            episode_rewards = torch.tensor(episode_rewards)
            episode_log_probs = torch.tensor(episode_log_probs, requires_grad=True)
            loss = self.agent.train(episode_log_probs, episode_rewards)
            #print(episode_actions)
            print("Episode=" + str(self.episode_no) + " Reward=" + str(total_reward) + " Loss=" + str(loss.item()) + " Current Pipes=" + str(passed_pipes) + " Max Pipes=" + str(self.agent.max_pipes))


learner = GameLearner(GameEnv(pipe_gap=150), Agent(number_of_frames, device, model_filename), render)
learner.reinforced_learning()
