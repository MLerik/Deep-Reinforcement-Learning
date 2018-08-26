# Report for training an RL-Agent on BananaBrain
## Contents
- [Overview Environment](#over)
- [Setting up the Agent](#set)
- [Training and Performance](#train)
- [Future Improvements](#future)

<a name="over"></a>
## Overview Environment
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

In this project I trained an agent to navigate (and collect bananas!) in a large, square world. As part of the Deep Reinforcement Learning Nano Degree @ Udacity.
Below you see a short sample gif of the environment as well as som details on how to get the Environment ready on your computer.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Thus the state is made up of binary dimensions and real valued dimension!

Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


<a name="set"></a>
## Setting up the Agent
### DQN - Agent
Given that the input state has both binary and real valued numbers and considering that discretizing this space would lead to a huge discrete state-space, I chose to implement the DQN agent using a [feed forward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) for Q-Valuefunction approximation.
#### Neural net as function approximator
The general idea when using neural networks for Q-Learning is to approximate the Q-Value of each action-state-configuration. The state of the environment is fed as input to the neural network and the network returns a Q-Value estimation for all possible actions.

Hence the input size to the neural network and the output size are given by the environment, in our case we have a 37 dimensional input and a 4 dimensional output.

The structure of the neural network inbetween input and output, the so called hidden layers has to be chose appropriately and is part of the engineering necessary to get good performance.

For the BananaBrain environment a neural network with 2 hidden layers with 64 hidden units each works well. The output of each layer (except the last layer) is passed through a non-linearity ([rectified linear unit in this case](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))).

To receive Q-Value estimations for a given state, a forward pass through the network is perfomed. In order to chose an action from these Q-Values we need a policy which the agent follows.
#### Epsilon-greedy policy


<a name="train"></a>
## Training and Performance

<a name="future"></a>
## Future Improvements
