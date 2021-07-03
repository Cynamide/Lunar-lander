import gym
import numpy as np
import tensorflow as tf


model =tf.saved_model.load("Breakout-test-model")
env = gym.make("LunarLander-v2")
state = env.reset().reshape(1,8)
rewar = 0
ep=0
while ep<10:
    env.render()
    pred = model(state)
    pred = np.argsort(pred)
    action = pred[0][3]
    #action = np.argmax(model.predict(state))
    next_state, reward, terminated, info = env.step(action)
    next_state = next_state.reshape(1,8)
    state = next_state
    rewar+=reward
    if terminated:
        env.reset()
        ep+=1
        print(rewar)
        rewar = 0

