from mnist_env import MNISTEnv
from actor_critic_agent import MNISTNet, ActorCriticNNAgent
import numpy as np
import time

def train(iterations, episodes, verbose=False):
    
    def obs_to_input(obs):
        # reshape to (1, 28, 28)
        return obs[np.newaxis, ...]
    
    # initialize agent
    agent = ActorCriticNNAgent(MNISTNet, obs_to_input=obs_to_input, df=0.1)
    
    # intialize environment
    env = MNISTEnv(type='train', seed=None)
    
    # training loop
    start = time.time()
    for iter in range(iterations):
    
        if iter % 10 == 0: print("Starting iteration %d" % iter)
        rewards = []
    
        # play out each episode
        for ep in range(episodes):
        
            if verbose and iter % 10 == 0 and ep == 0:
                display = True
                
            else:
                display = False
            
            observation = env.reset()
            agent.new_episode()
            total_reward = 0
            
            done = False
            while not done:
                    
                action = agent.act(observation, env, display=display)
                observation, reward, done, info = env.step(action)
                
                if display: print("Actual reward:", reward)
                agent.store_reward(reward)
                total_reward += reward
            
            rewards.append(total_reward)
            
            if display: env.render()
            
        # adjust agent parameters based on played episodes
        agent.update()
        
        # print performance for this iteration
        if iter % 10 == 0:
            print("Mean total reward / episode: %.3f" % np.mean(rewards))
        
    end = time.time()
    print("Completed %d iterations of %d episodes in %.3f s" % \
          (iterations, episodes, end - start))
        
    # return trained agent
    return agent
    
    
def eval(agent, n_test=1000):
    # evaluate a trained agent
    
    env = MNISTEnv(type='test', seed=None)
    
    observation = env.reset()
    
    done = False
    while not done:
        
        action = agent.act(observation, env, display=True)
        observation, reward, done, info = env.step(action)
        
        print("Received reward %.1f on step %d" % (reward, env.steps))
        env.render()
    
    
def test(agent, n_test=1000):
    # calculate test average reward
    
    env = MNISTEnv(type='test', seed=None)
    
    rewards = []
    for _ in range(n_test):
    
        observation = env.reset()
        total_reward = 0
        
        done = False
        while not done:
                
            action = agent.act(observation, env, display=False)
            observation, reward, done, info = env.step(action)
            
            total_reward += reward
        
        rewards.append(total_reward)
        
    print("Mean total reward / episode: %.3f" % np.mean(rewards))
    
    
if __name__ == '__main__':
    
    print("Training...")
    trained_agent = train(200, 100, verbose=False)
    test_agent = trained_agent.copy()
    
    print("Testing...")
    test(test_agent)
    
    print("Evaluating...")
    for _ in range(5):
        eval(test_agent)