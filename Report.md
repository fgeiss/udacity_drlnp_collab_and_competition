# Report 

## Baseline Evaluation
To be able to evaluate the performance of a trained agent we performed a baseline test at the beginning of the project and run the environment N=100 times with random uniformly distributed actions. Below diagram shows the distribution with its mean and standard deviation from this experiment.

![Baseline Tennis](tennis_baseline.png)

## Algorithm
We implemented a Multi Agent Deep Deterministic Policy Gradient (MADDP) algorithm. This algorithm is described in detail in [this landmark paper](https://arxiv.org/abs/1706.02275). Key idea is the utilization of multiple actor-critic agents, in our case 2, where the critic networks are centralized, i.e. they take as inputs the actions and observations of all agents. The actor network takes only the observation of the respective agent as input. 

![Diagram](maddpg.png)

This core part is implemented in the classes SingleAgent and MultiAgent. The actor and critic networks for both agents are identical, respectively and consist of two hidden layers with 100 neurons.


## Learning

### Results
The goal of a score larger than 0.5 was achieved after approx. 3600 episodes. Due to the non-stationarity of the setup the average score over episodes does not follow a continuously increasing curve but rather stays almost constant zero for the first 1500 episodes end then increases significantly to approx. 0.1. At this point the players learned to successfully hit the first ball of the game. After approx. 3400 episodes there is another significant increment that reflect the point at which the agents learned to hit back balls from the other agents.

![Learning](tennis_learning_hits.png)

With the trained agents we performed the baseline experiment of 100 episodes again. The results are shown below.

![Trained_Agents](tennis_solved.png)
