# Report for MARL Training on Tennis environment using DDPG
## Contents
- [Overview Environment](#over)
- [Actor-Critic Model](#qlearning)
- [Setting up the Agent](#set)
- [Training and Performance](#train)
- [Future Improvements](#future)

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"




### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


<a name="set"></a>
## Setting up the Agent

### DDPG - Agent

Because we have a continuous action and state space it makes sence to use a policy based method. Starting from the [example](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) from the Udacity Deep Reinforcement Learning Nano Degree, we only need to make minor adjustments to get a good performing agent.

First adjustments are of course the state space and action space size, and the less straight forward adjustment is the modification to support 2 agents. I implemented two different approaches for training.

Information on how to implement a DDPG-Agent can be found [here](https://arxiv.org/abs/1509.02971)

#### Homogeneous Agents
In this approach I assume the 2 agents to be copies of each other. In other words we only need to [implement one DDPG-Agent](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Tennis/Agent/ddpg_agent_homogeneous.py) and just let it give actions to each individual state. If you combine this with one shared replay buffer, what you get is a rudimentary parallelization of training. These 2 agents explore 2 trajectories in parallel using the same policy.
This approach was very successfull and the task was solved after **177 episodes**. It however has the drawback that the replay buffer only contains trajectories of one policy and thus exploration is not optimal.


#### Heterogeneous Agents
In this approach I assume the 2 agents to be individuals. [Here](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Tennis/Agent/ddpg_agent.py) I implemented a seperate neural network for each agent to represent each agents policy and only shared the replay buffer between all agents. With this approach we collect experiences from many different policies at the same time. Unfortunately I don't know how well this would behave because training was very slow and I had to abort before the task was solved. I hope to let a training run for longer times to see how it hold up.

#### Learning through self play
To further improve stability of training, I let the agent performe so called self play where I only updated one of the two agents on a regular basis. The second agent was a copy of the first agent, which received a new copy every 20 episodes. In this way the policy of one agent stayed fixed during 20 trials, which produced slower but more stable learning.

#### Final network structure
I used two identical network setups for the actor and critic, except for the output layer. The actor outputs 4 values in [-1,1] whereas the critic only outputse a single value representing the state-action-value.

Layer | Actor | Critic
------------ | ------------ | -------------
Input | 24 | 24
Hidden 1 | 400 | 400
Hidden 2 | 300 | 300
Output | 2 | 1

<a name="train"></a>
## Training and Performance
Training was performed using the [Jupyter Notebook file](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Tennis/Tennis.ipynb) and can run locally on a decent laptop in less than 2 hours. Convergance and stability of this approach were very good.

### Hyperparameters
~~~~
- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 1024  # minibatch size
- GAMMA = 0.99  # discount factor
- TAU = 1e-3  # for soft update of target parameters
- LR_ACTOR = 5e-4  # learning rate of the actor
- LR_CRITIC = 1e-3  # learning rate of the critic
- WEIGHT_DECAY = 0  # L2 weight decay
- LEARN_DELAY = 20  # Collect memories for n steps
- LEARN_STEPS = 10  # Do n number of weight updates
- OU-Noise MU = 0. # Each agent has an OU-Noise-Process for each of its possible actions (4)
- Ou-Noise Sigma = 0.1 
~~~~

### Performance

The environment was solved after **177 Episodes** but this was not the converging point of training. Letting the agents train for even longer showed the difficulty in convergence with multi agent reinforcement learning, where performance sometime became worse with more traiing.
The trained Networks can be found in the [/Nets](https://github.com/androiddeverik/Deep-Reinforcement-Learning/tree/master/Tennis/Nets) folder.

The figure below shows the average score over the last 100 epsiodes. 

[image6]:https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Tennis/Images/Training.png
![Training][image6]


<a name="future"></a>
## Future Improvements
Future improvements would consist of testing the heterogeneous approach, using different actor-critic-models. A very intersting task would also be to learn this task from pixels, as this to me seems like a much harder task where the agent needs to learn 2D projections of the 3D world.

