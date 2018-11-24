[//]: # (Image References)

[image1]: https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Images/reacher.gif "Environment"

# Project 1: Navigation

### Introduction

In this project I trained an multiple agents to solve the reacher task as part of the Deep Reinforcement Learning Nano Degree @ Udacity.
Below you see a short sample gif of the environment as well as som details on how to get the Environment ready on your computer.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The barrier for solving the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).

### Getting Started

1. The environment has been included in this repository for convenience.
2. Paths in the Continuous_Control.nb file have to be updated to point to the correct directories.

### Instructions

Follow the instructions in [`Continuous_Control.ipynb`](https://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/Continuous_Control.ipynb) to see how the agent performs during and after training.
Read the ['Report'](hhttps://github.com/MLerik/Deep-Reinforcement-Learning/blob/master/Continuous_Control/report.md) to see details about the chosen model, hyperparameters and additional considerations
