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

Actor-Critic-Models fall in the class between policy-based and value-based model. Models of this class take advantage from both different approaches.
- **State**: A 33 Dimensional array representing position, velocity, acceleration and further information about the target position
- **Action**: 4 Floats in the range [-1,1] representing the torque at each joint of the robot arm
- **Reward**: Reward returned by the environment which depends on the state. It can take the values (0,0.1)
- **Next State**: A 33 dimensional array containing all the available information about the state of the environment after the action was executed
- **Done** : True or False depending whether the episode has terminated or not. If Done is true there is no next state and the reward is not discounted!



<a name="set"></a>
## Setting up the Agent

### DDPG - Agent

Because we have a continuous action and state space it makes sence to use a policy based method. Starting from the [example](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) from the Udacity Deep Reinforcement Learning Nano Degree, we only need to make minor adjustments to get a good performing agent.

First adjustments are of course the state space and action space size, and the less straight forward adjustment is the modification to support 20 agents. I implemented two different approaches for training.

Information on how to implement a DDPG-Agent can be found [here](https://arxiv.org/abs/1509.02971)
#### Homogeneous Agents
In this approach I assume all the 20 agents to be copies of each other. In other words we only need to [implement one DDPG-Agent](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Agent/ddpg_agent.py) and just let it give actions to each individual state. If you combine this with one shared replay buffer, what you get is a rudimentary parallelization of training. These 20 agents explore 20 trajectories in parallel using the same policy.
This approach was very successfull and the task was solved after **177 episodes**. It however has the drawback that the replay buffer only contains trajectories of one policy and thus exploration is not optimal.

#### Heterogeneous Agents
In this approach I assume all the 20 agents to be individuals. [Here](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Agent/ddpg_agent.py) I implemented a seperate neural network for each agent to represent each agents policy and only shared the replay buffer between all agents. With this approach we collect experiences from many different policies at the same time. Unfortunately I don't know how well this would behave because training was very slow and I had to abort before the task was solved. I hope to let a training run for longer times to see how it hold up.

#### Final network structure
I used two identical network setups for the actor and critic, except for the output layer. The actor outputs 4 values in [-1,1] whereas the critic only outputse a single value representing the state-action-value.

Layer | Actor | Critic
------------ | ------------ | -------------
Input | 33 | 33
Hidden 1 | 400 | 400
Hidden 2 | 300 | 300
Output | 4 | 1

<a name="train"></a>
## Training and Performance
Training was performed using the [Jupyter Notebook file](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Continuous_Control.ipynb) and can run locally on a decent laptop in less than 2 hours. Convergance and stability of this approach were very good.

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

The environment was solved after **177 Episodes** but this was not the converging point of training. Letting the agents train for even longer periods of times showed significant improvement up to more than **35 point average** of reward.
The trained Networks can be found in the [/Nets folder](https://github.com/androiddeverik/Deep-Reinforcement-Learning/tree/master/Navigation/Nets).


[image6]:https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Images/Training.png
![Training][image6]


<a name="future"></a>
## Future Improvements
Future improvements would consist of testing the heterogeneous approach, using different actor-critic-models. A very intersting task would also be to learn this task from pixels, as this to me seems like a much harder task where the agent needs to learn 2D projections of the 3D world.

## UnityML Environment
To me it seemed overly complicated to use UnityML environment to train a task with only 33 dimensional input. Modeling a 2-joint robotic arm can easily be done using just numpy. My intuition would be that such an environment would perform much better than UnityML and thus training could be done much more efficiently. Given that reinforcement learning is still very sample inefficient I believe that more work should be invested in good simulation environments with high performance.
