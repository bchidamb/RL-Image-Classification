from mnist_env import MNISTEnv
import numpy as np

# Run this file to test if the environment is working

env = MNISTEnv(type='train', seed=None)

obs = env.reset()
done = False

while not done:

    env.render()
    dir, Y_pred = env.action_space.sample()
    print("Agent moved %s" % (['North', 'South', 'East', 'West'][dir]))
    print("Agent guessed %d" % Y_pred)
    
    _, reward, done, _ = env.step(np.array([dir, Y_pred]))
    print("Received reward %.1f on step %d" % (reward, env.steps))