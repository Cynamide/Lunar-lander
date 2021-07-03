import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import cv2
import keyboard
#from atari_wrapper import wrap_deepmind
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import time

reward_list = [0]
batch_size = 128
num_of_episodes = 10_000
show_preview = 2000 
env = gym.make("LunarLander-v2")
#env=wrap_deepmind(env)
done=False
env.reset()
update_weights_counter = 5
class Agent:

    def __init__(self, enviroment):

        self.action_size = enviroment.action_space.n
        self.ob_size = 8 #check with enviroment.observation_space
        self.deque_maxlen = 1_000_000
        self.expirience_replay = deque(maxlen=self.deque_maxlen)
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.discount = 0.99
        self.max_timesteps = 3000
        self.loss = 0
        self.loss_list = []
       
        #self.total_frames = 100 #10_000_000
        self.max_episodes = 500#20000   
        #self.learn_every_n_frame = 1
        self.explore_episodes = 2#1000 
        #self.update_target_every = 1 
        #self.min_epsilon_episode = 1149#2000
        self.epsilon_decrement = 0.996#(self.epsilon - self.epsilon_min) / (self.min_epsilon_episode-self.explore_episodes)
        

        self.q_model = self.atari_model()
        #self.target_q_model = self.atari_model()
        #self.align_target_model()

    def act(self, state, enviroment,episode_no):
        if episode_no < self.explore_episodes:
            return enviroment.action_space.sample()
            
        elif np.random.rand() <= self.epsilon:
           return enviroment.action_space.sample()
        else:
           q_values = self.q_model.predict(state)
           return np.argmax(q_values[0])

    def store(self, state, action, reward, next_state,terminated):
        if len(self.expirience_replay) == self.deque_maxlen:
           self.expirience_replay.popleft() 
           self.expirience_replay.append((state, action, reward, next_state,terminated))
        else:
           self.expirience_replay.append((state, action, reward, next_state,terminated))
           

    #def align_target_model(self):
        #self.target_q_model.set_weights(self.q_model.get_weights())

    def retrain(self, batch_size):
       
        minibatch = random.sample(self.expirience_replay, batch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.q_model.predict(current_states.squeeze())
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.q_model.predict(new_current_states.squeeze())

        x=[]
        y=[]
        
            # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state,terminated) in enumerate(minibatch):
            #reward = self.transform_reward(reward)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            max_future_q = np.max(future_qs_list[index])
            # Update Q value for given state
            current_qs = current_qs_list[index]
            if terminated:
                current_qs[action] = reward
            else:
                current_qs[action] = reward + agent.discount * max_future_q 
            # And append to our training data
            x.append(current_state.squeeze())
            y.append(current_qs)
        # Fit on all samples as one batch, log only on terminal state
        history = self.q_model.fit(np.array(x), np.array(y), batch_size=batch_size,epochs = 1, verbose=0)
        self.loss += history.history['mse'][0]

    def atari_model(self):
        img_size = (84,84,4)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(150, activation='relu',input_shape = [self.ob_size]),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
        optimizer = tf.keras.optimizers.Adam(0.0005)
        model.compile(optimizer, loss='mse', metrics=['mse'])
        return model

agent = Agent(env)
frame = 0
ep_reward = 0
realign = 1
exit = False
for ep in range(1,agent.max_episodes+1):
  if exit == True:
      break
  begin = time.time()
  state = env.reset()
  for i in range(agent.max_timesteps):
    state = state.reshape(1,8)
    action = agent.act(state, env,ep)
    next_state, reward, terminated, info = env.step(action)
    next_state = next_state.reshape(1,8)
    agent.store(state, action, reward, next_state,terminated)
    ep_reward += reward
    state = next_state
    if terminated:
        if ep % 10 == 0: #and frame > agent.explore_frames:
            print("episode NO: ",ep)
            print("REWARD: ",reward_list[-1])
            print("replay size",len(agent.expirience_replay))
        if len(reward_list)>0:
            reward_list.append((reward_list[-1]*len(reward_list)+ep_reward)/(len(reward_list)+1))
        else:
            reward_list.append(ep_reward)
        #reward_list.append(ep_reward)
        if np.sum(reward_list[-10:])/10 >200:
            exit =True
        ep_reward = 0
        break
        #if episode == agent.max_episodes:
            #print("Done processing ",agent.max_episodes,"episodes")
           # break
        
    else:
        
        if ep > agent.explore_episodes:
            agent.retrain(batch_size)
            #if realign % agent.update_target_every == 0:
                #agent.align_target_model()
               
            #realign +=1
            agent.epsilon = (agent.epsilon * agent.epsilon_decrement) if agent.epsilon > agent.epsilon_min else agent.epsilon_min 

  agent.loss_list.append((np.sum(agent.loss_list)+agent.loss)/(len(agent.loss_list)+1))
  end = time.time()
  print(end-begin)

       
            
            

     
agent.q_model.save("Breakout-test-model")     
reward_list=np.array(reward_list)
np.save("Rewards",reward_list)

epochs = range(1,agent.max_episodes+1)
plt.plot(epochs,reward_list[1:],'b',label='Reward distribution')
plt.title('Reward Analysis')
plt.legend()
plt.figure()
plt.plot(epochs,agent.loss_list,'b',label='Reward distribution')
plt.title('Loss Analysis')
plt.legend()
plt.show()
