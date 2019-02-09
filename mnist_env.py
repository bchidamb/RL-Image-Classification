import gym
from gym import error, spaces, utils
from gym.utils import seeding

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

    -- need to determine threshold of correct classification

'''

class MNISTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, type='train', seed=2069):
        # TODO
        pass
        
    def step(self, action):
        # action consists of
        #   1. direction in {N, S, E, W}
        #   2. prediction over possible classes (0-9)
        # state consists of masked image (28x28)
        
        # TODO
        pass
        
    def reset(self):
        # TODO
        pass
        
    def render(self, mode='human', close=False):
        # TODO
        pass