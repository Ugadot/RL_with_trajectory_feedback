import gym
import time

env = gym.make("Hopper-v2")
env.reset()

time.sleep(10)
ac = env.action_space
for i in range(5):
    o, r, d, i = env.step(ac.sample())
    print(o)
    print(r)
    print(d)
    print(i)
    
for i in range(100000):
    #time.sleep(1)
    env.step(ac.sample())
    env.render()