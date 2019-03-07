import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

MAX_STEPS = 20
WINDOW_SIZE = 7
RANDOM_LOC = False

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

    -- for now, agent outputs direction of movement and class prediction (0-9)
    -- correct guess ends game
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
            
        h, w = self.X[0].shape
        self.h = h // WINDOW_SIZE
        self.w = w // WINDOW_SIZE
        
        self.mask = np.zeros((h, w))
        
        # action is an integer in {0, ..., 39}
        # see 'step' for interpretation
        self.action_space = spaces.Discrete(40)
        self.observation_space = spaces.Box(0, 255, [h, w])
        
    def step(self, action):
    
        # action a consists of
        #   1. direction in {N, S, E, W}, determined by = a (mod 4)
        #   2. predicted class (0-9), determined by floor(a / 4)
        assert(self.action_space.contains(action))
        dir, Y_pred = action % 4, action // 4
        
        self.steps += 1
        
        move_map = {
            0: [-1, 0], # N
            1: [1, 0],  # S
            2: [0, 1],  # E
            3: [0, -1]  # W
        }
        
        # make move and reveal square
        self.pos = np.clip(self.pos + move_map[dir], 0, [self.h-1, self.w-1])
        self._reveal()
        
        # state (observation) consists of masked image (h x w)
        obs = self._get_obs()
        
        # -0.1 penalty for each additional timestep
        # +1.0 for correct guess
        reward = -0.1 + int(Y_pred == self.Y[self.i])
        
        # game ends if prediction is correct or max steps is reached
        done = Y_pred == self.Y[self.i] or self.steps >= MAX_STEPS
        
        # info is empty (for now)
        info = {}
        
        return obs, reward, done, info
        
    def reset(self):
        # resets the environment and returns initial observation
        # zero the mask, move to random location, and choose new image
        
        # initialize at random location or image center
        if RANDOM_LOC:
            self.pos = np.array([np.random.randint(self.h), 
                                 np.random.randint(self.w)])
        else:
            self.pos = np.array([int(self.h / 2), int(self.w / 2)])
            
        self.mask[:, :] = 0
        self._reveal()
        self.i = np.random.randint(self.n)
        self.steps = 0
        
        return self._get_obs()
        
    def _get_obs(self):
        obs = self.X[self.i] * self.mask / 255
        assert self.observation_space.contains(obs)
        return obs
        
    def _reveal(self):
        # reveal the window at self.pos
        h, w = self.pos
        h_low, h_high = h * WINDOW_SIZE, (h + 1) * WINDOW_SIZE
        w_low, w_high = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
        
        self.mask[h_low:h_high, w_low:w_high] = 1
        
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