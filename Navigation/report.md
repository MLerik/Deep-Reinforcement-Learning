# Report for training an RL-Agent on BananaBrain
## Contents
- [Overview Environment](#over)
- [Q Learning](#qlearning)
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


<a name="qlearning"></a>
## Q Learning
Definition:

[Q-learning](https://en.wikipedia.org/wiki/Q-learning) is a reinforcement learning technique used in machine learning. The goal of Q-Learning is to learn a policy, which tells an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards, without requiring adaptations.

For any finite Markov decision process (FMDP), Q-learning finds a policy that is optimal in the sense that it maximizes the expected value of the total reward over all successive steps, starting from the current state. Q-learning can identify an optimal action-selection policy for any given FMDP, given infinite exploration time and a partly-random policy. "Q" names the function that returns the reward used to provide the reinforcement and can be said to stand for the "quality" of an action taken in a given state

The aim of this algorithm is to maximise the future discounted reward. This means that we want to maximize sum over all future rewards. To achieve this we can perform the following update to our Q function:

[image4]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/q_learning.svg
![qupdate][image4]

To perform this Q-Learning the agent needs to collect experiences, which means that the agent needs to interact with the environment. The interaction with the environment causes the environment to change and the agent receives an update about the new state of the environment as well as a reward depending on that state. Such a tuple for learning contains the following data

- **State**: A 37 dimensional array containing all the available information about the current state of the environment
- **Action**: An integere in the range [0,4] representing the action the agent took
- **Reward**: Reward returned by the environment which depends on the state. It can take the values (-1,0,1)
- **Next State**: A 37 dimensional array containing all the available information about the state of the environment after the action was executed
- **Done** : True or False depending whether the episode has terminated or not. If Done is true there is no next state and the reward is not discounted!



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
The so called epsilon-greedy policy is used to chose an action from the given Q-Value approximations. In this policy the agent will choose the action with the highest Q-Value with a probability of P(a(maxQ) = 1-epsilon and a random action with the probability of epsilon. This policy is good for training as it can be shown that it will converge to the optimal policy.

#### Enhancements to basic DQN Network: Dueling DQN
[image2]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/dueling.png

In order to improve the performance of the DQN Network a minor addition can be done. By splitting the Q-Function into its to parts, namely state value estimation and action value estimation we can improve the behavior of the agent.
![Dueling][image2]

This is specially helpful when we have many states where no action leads to a better state. In these cases the split of the estimations helps, because the value of each state can be considered independently of the possible actions.

This improvement can easily be implemented by splitting the neural network into two seperate streams. Both streams take the state as an input and are merged before the output. For merging we sum the action values with the state value and subtract the mean of the action values.


#### Enhancements to basic DQN Network: Double DQN
To reduce the problem of the moving target we can further enhance our network by seperating the prediction of the Q-Values and the Q-Value target generation. In other words we introduce a copy of the DQN network which has a slower weight update than our main network. We use this network to predict the Q-Values for the chosen actions instead of predicting the Q-Values with the same network which chooses the action.

This improvement is simple implemented by generating a copy of the neural network and using this copied network to predict the Q-values. Furthermore an update strategy for this copied network needs to be implemented. In my case I chose to update the weights as a convex combination of the current network weights and the network weights of the trained network. In this case we end up with a slowly evovling network which keeps our Q-Value targets more stable during training.

#### Final network structure
[image3]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/neural_net.png
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
All the different implementations were compared. The best result (fastest meeting required 13+ point average) was achieved with the dueling DQN which reached the target after **239 Episodes**. 


[image6]:https://github.com/androiddeverik/Deep-Reinforcement-Learning/blob/master/Navigation/Training_Final.png
![Training][image6]


<a name="future"></a>
## Future Improvements
