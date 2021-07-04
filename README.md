# Lunar-lander
A Deep Q - Learning Model for training on Lunar Lander environment provided by OpenAI Gym
<p align="center"><img src="https://i.imgur.com/bwMsTiK.gif"> </p>  
 
# Description 
This is my [TensorFlow](https://www.tensorflow.org/) implementations of Deep Q learning model proposed in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- The task is to land the space-ship between the flags smoothly. The ship has 3 throttles in it. One throttle points downward and other 2 points in the left and right direction. With the help of these, you have to control the Ship. There are 2 version for this task. One is discrete version which has discrete action space and other is continuous which has continuous action space.
- In order to solve the episode you have to get a reward of +200 for 100 consecutive episodes. I solved both the version under 400 episodes.
- This environment was in my capstone project as part of my [University of Alberta Specialization](https://coursera.org/share/5f2b2e39b21fc68ac4062f08ad6e87d3) which did not use Tensorflow to solve the Environment.

## Prerequisites

- Python 3.7
- [OpenAI GYM](https://gym.openai.com/)
- [Tensorflow 2.4.x](https://github.com/tensorflow/tensorflow/)
- [NumPy](http://www.numpy.org/)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)

## Running the Script
- To run the script :
	```bash
	python lunar_lander.py
	```
- Make sure to change the variable values according to your needs after running the first time to see different results.
- To test the model trained simply run :
	```bash
	python test_model.py
	```
