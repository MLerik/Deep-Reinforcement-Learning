# Report for training an DDPG Agent to solve the Reacher Task
## Contents
- [Overview Environment](#over)
- [Actor-Critic Model](#qlearning)
- [Setting up the Agent](#set)
- [Training and Performance](#train)
- [Future Improvements](#future)

[image1]: https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Images/reacher.gif "Environment"
<a name="over"></a>
## Overview Environment
In this project I trained an multiple agents to solve the reacher task as part of the Deep Reinforcement Learning Nano Degree @ Udacity.
Below you see a short sample gif of the environment as well as som details on how to get the Environment ready on your computer.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The barrier for solving the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).


<a name="qlearning"></a>
## Actor-Critic Model
### Deep Deterministic Policy Gradients (DDPG)
[actor-critic]:https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Images/A-regular-actor-critic-model-TD-temporal-difference.png
![architecture][actor-critic]

Actor-Critic-Models fall in the calss between policy-based and value-based model. Models of this class take advantege from both different approaches.
- **State**: A 33 Dimensional array representing position, velocity, acceleration and further information about the target position
- **Action**: 4 Floats in the range [-1,1] representing the torque at each joint of the robot arm
- **Reward**: Reward returned by the environment which depends on the state. It can take the values (0,0.1)
- **Next State**: A 33 dimensional array containing all the available information about the state of the environment after the action was executed
- **Done** : True or False depending whether the episode has terminated or not. If Done is true there is no next state and the reward is not discounted!



<a name="set"></a>
## Setting up the Agent

### DDPG - Agent

Given that the input state has both binary and real valued numbers and considering that discretizing this space would lead to a huge discrete state-space, I chose to implement the DQN agent using a [feed forward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) for Q-Valuefunction approximation. Many functions such as memory, experience replay, step and so on were taken from the exercise code from the Udacity Nano Degree program.

#### Neural net as function approximator

The general idea when using neural networks for Q-Learning is to approximate the Q-Value of each action-state-configuration. The state of the environment is fed as input to the neural network and the network returns a Q-Value estimation for all possible actions.

Hence the input size to the neural network and the output size are given by the environment, in our case we have a 37 dimensional input and a 4 dimensional output.

The structure of the neural network inbetween input and output, the so called hidden layers has to be chose appropriately and is part of the engineering necessary to get good performance.

For the BananaBrain environment a neural network with 2 hidden layers with 64 hidden units each works well. The output of each layer (except the last layer) is passed through a non-linearity ([rectified linear unit in this case](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))).

To receive Q-Value estimations for a given state, a forward pass through the network is perfomed. In order to chose an action from these Q-Values we need a policy which the agent follows.


#### Final network structure
[image3]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/figs/neural_net.png
![Network][image3]

The image above illustrates the final network structure used for training and the table below shows the specific implementation and chosen parameters for each subnet.

Layer | Action Value Net | State Value net
------------ | ------------ | -------------
Input | 37 | 37
Hidden 1 | 64 | 64
Hidden 2 | 32 | 32
Output | 4 | 1



<a name="train"></a>
## Training and Performance
Training was performed using the [Jupyter Notebook file](https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/Navigation.ipynb) and all different implementations were compared to each other. For all implementations I used the same hyperparameters in order to be able to compare the state. For better comparison one would need to reset all the random number seeds of the agent and the environment to the same seed. 

### Hyperparameters
- Learning Rate = 1e-5
- Batch Size = 64
- Discount factor gamma = 0.99
- Soft update factor Tau = 1.e-3
- Initial epsilon = 0.1
- Minimal epsioln = 0.001
- Epsilon decay = 0.99

### Performance
All the different implementations were compared. The best result (fastest reaching required 13+ point average) was achieved with the dueling DQN which reached the target after **239 Episodes**. In general we can observe that double DQN brings less improvement than the dueling architecture. What we observe is that the performance of the different approaches vary strongly on a trial by trial basis. This is to be expected as the algorithm with a stochastic gradient. Even though all algorithms will converge to the optimal policy, the stochastic gradients do not guarantee that the fastest path to the optimal strategy will be followed. Hence to habe good comparison between the different implementations one needs to guarantee that the same environmental steps are shown to all agents. This of course is not possible in our setting and thus a comparison needs to be done over many trials. This was not done in this project due to large computational time needed.

The trained Networks can be found in the [/Nets folder](https://github.com/androiddeverik/Deep-Reinforcement-Learning/tree/master/Navigation/Nets).


[image6]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/figs/Training_Final.png
![Training][image6]


<a name="future"></a>
## Future Improvements
To further imrpove the already good results there are many different ideas and I would like to highlight two of them here.
### Hierarchical RL
In [hierarchical RL](https://arxiv.org/abs/1604.06057) we can divide the full task into smaller subtasks. In this case here we could consider the task to consist of two sub-objectives.
#### Searching
First we have to find the next yellow banana that we want to collect. This task can be optimized to find the best reachable banana from the current position. A scanning behavior would probably be helpful here.
#### Navigation
Once we have found a suitable banana, we need to take the shortest path to the banana without collecting any blue bananas.

In hierarchical RL we could train two networks to perfom the above task very well by themselfes. In a next step we then train an agent to decide when to do searching and when to do navigating. For some complex tasks it has been shown to improve performance if the task is divided into subtasks.


### RAINBOW
A much simpler improvement to the current model would be to implement even more enhancement to the DQN algorithm. One example would be to implement the full [RAINBOW](https://arxiv.org/pdf/1710.02298.pdf) algorithm
