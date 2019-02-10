from mnist_env import MNISTEnv
import numpy as np

env = MNISTEnv(type='train', seed=None)

obs = env.reset()
done = False

while not done:

    env.render()
    dir, pred_bit, Y_pred = env.action_space.sample()
    print("Agent moved %s" % (['North', 'South', 'East', 'West'][dir]))
    print("Agent guessed %d" % Y_pred if pred_bit else "Agent did not guess")
    
    _, reward, done, _ = env.step(np.array([dir, pred_bit, Y_pred]))
    print("Received reward %.1f on step %d" % (reward, env.steps))