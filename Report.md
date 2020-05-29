# Report 

## Baseline Evaluation
To be able to evaluate the performance of a trained agent we performed a baseline test at the beginning of the project and run the environment N=100 times with random uniformly distributed actions. Below diagram shows the distribution with its mean and standard deviation from this experiment.

![Baseline Tennis](tennis_baseline.png)

## Algorithm
We implemented a Multi Agent Deep Deterministic Policy Gradient (MADDP) algorithm. This algorithm is described in detail in [this landmark paper](https://arxiv.org/abs/1706.02275). Key idea is the utilization of multiple actor-critic agents, in our case 2, where the critic networks are centralized, i.e. they take as inputs the actions and observations of all agents. The actor network takes only the observation of the respective agent as input. This core part is implemented in the classes SingleAgent and MultiAgent.

## Learning

![Learning](learning_multi_ddpg.png)

![Learning](tennis_learning_hits.png)

![Trained_Agents](tennis_solved.png)
