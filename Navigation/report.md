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

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


<a name="set"></a>
## Setting up the Agent
### DQN - Agent

<a name="train"></a>
## Training and Performance

<a name="future"></a>
## Future Improvements
