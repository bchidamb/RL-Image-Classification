from mnist_env import MNISTEnv
import numpy as np

# Run this file to test if the environment is working

env = MNISTEnv(type='train', seed=None)

obs = env.reset()
done = False

while not done:

    env.render()
    action = env.action_space.sample()
    dir, Y_pred = action % 4, action // 4
    print("Agent moved %s" % (['North', 'South', 'East', 'West'][dir]))
    print("Agent guessed %d" % Y_pred)
    
    _, reward, done, _ = env.step(action)
    print("Received reward %.1f on step %d" % (reward, env.steps))