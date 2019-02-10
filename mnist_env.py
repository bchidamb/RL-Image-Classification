import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

'''
Notes:
    Agent navigation for image classification. We propose
    an image classification task starting with a masked image
    where the agent starts at a random location on the image. It
    can unmask windows of the image by moving in one of 4 directions: 
    {UP, DOWN, LEFT, RIGHT}. At each timestep it
    also outputs a probability distribution over possible classes
    C. The episode ends when the agent correctly classifies the
    image or a maximum of 20 steps is reached. The agent receives a 
    -0.1 reward at each timestep that it misclassifies the
    image. The state received at each time step is the full image
    with unobserved parts masked out.

    -- for now, agent outputs just the class prediction (0-9) and a prediction bit
    -- agent's guess only counts if bit is set to 1
    -- incorrect guess is penalized with reward -1
    -- correct guess ends game
    
    -- also, 20 step maximum seems harsh, may want to increase
'''

class MNISTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, type='train', seed=2069):
        
        if seed:
            np.random.seed(seed=seed)
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        if type == 'train':
            self.X = x_train
            self.Y = y_train
            self.n = len(y_train)
            
        elif type == 'test':
            self.X = x_test
            self.Y = y_test
            self.n = len(y_test)
            
        self.h, self.w = self.X[0].shape
        
        # action is a 3-tuple in {0, 1, 2, 3} x {0, 1} x {0, ..., 9}
        # see 'step' for interpretation
        self.action_space = spaces.MultiDiscrete([4, 2, 10])
        self.observation_space = spaces.Box(0, 255, [self.h, self.w])
        
    def step(self, action):
    
        # action consists of
        #   1. direction in {N, S, E, W}
        #   2. whether to make prediction (0/1)
        #   2. predicted class (0-9)
        assert(self.action_space.contains(action))
        dir, pred_bit, Y_pred = action
        
        self.steps += 1
        
        move_map = {
            0: [-1, 0], # N
            1: [1, 0],  # S
            2: [0, 1],  # E
            3: [0, -1]  # W
        }
        
        # make move and reveal square
        self.pos = np.clip(self.pos + move_map[dir], 0, [self.h-1, self.w-1])
        self.mask[tuple(self.pos)] = 1
        
        # game ends if prediction is correct or max steps is reached
        done = pred_bit * (Y_pred == self.Y[self.i]) or self.steps >= 20
        
        # -0.1 penalty for each additional timestep
        # -1.0 penalty for incorrect guess
        reward = -0.1 - pred_bit * (Y_pred != self.Y[self.i])
        
        # state (observation) consists of masked image (h x w)
        obs = self.X[self.i] * self.mask
        assert self.observation_space.contains(obs)
        
        # info is empty (for now)
        info = {}
        
        return obs, reward, done, info
        
    def reset(self):
        # resets the environment and returns initial observation
        # zero the mask, move to random location, and choose new image
        
        self.pos = np.array([np.random.randint(self.h), 
                             np.random.randint(self.w)])
        self.mask = np.zeros((self.h, self.w))
        self.mask[tuple(self.pos)] = 1
        self.i = np.random.randint(self.n)
        self.steps = 0
        
        obs = self.X[self.i] * self.mask
        return obs
        
    def render(self, mode='human', close=False):
        # display mask, full image, and masked image
        
        plt.figure(figsize=(10, 3))
        plt.suptitle("Step %d" % self.steps)
        
        plt.subplot(131)
        plt.imshow(255 * self.mask)
        plt.subplot(132)
        plt.imshow(self.X[self.i])
        plt.subplot(133)
        plt.imshow(self.X[self.i] * self.mask)
        
        plt.show()