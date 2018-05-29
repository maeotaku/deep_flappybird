import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pickle
import os.path


from policy import *

class Agent():

    def __init__(self, number_of_frames, device, path=None):
        self.UP = 119
        self.NONE = None
        self.Y_UP = 0
        self.Y_NONE = 1
        self.model = self.load(path)
        if self.model is None:
            self.model = Policy(number_of_frames).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.085, momentum=0.5)

    def load(self, filename):
        if not filename is None and os.path.exists(filename):
            print("Loading model...")
            infile = open(filename,'rb')
            model = pickle.load(infile)
            infile.close()
            print("Model loaded :D!")
            return model
        return None

    def save(self, filename):
        print("Writing model to disk...")
        outfile = open(filename,'wb')
        pickle.dump(self.model, outfile)
        outfile.close()

    def pick_action(self, x):
        x = torch.from_numpy(x).float()#.unsqueeze(0)
        probs = self.model.forward(x)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        prob = action.item()
        #if np.random.uniform() > prob:
        #    action = 0
        #else:
        #    action = 1
        return prob, log_prob, action

    def translate_o_to_action(self, o):
        if o==0:
            return self.UP
        return self.NONE

    def train(self, log_probs, rewards):
        rewards_applied = -log_probs.float() * rewards.float()
        loss = rewards_applied.sum()
        loss.backward()
        self.optimizer.step()
        return loss
