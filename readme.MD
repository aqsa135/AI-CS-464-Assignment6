### Assignment 6: Markov Decision Processes and Reinforcement Learning

#### Due Wed Nov 29, 11:59pm.

##### 100 points.

Note: please make sure that your name is someplace in your assignment! This is especially true if it's not obvious from your GitHub handle.

To turn in: please submit everything to your github repo. For the written portion of the assignment, please add a document to your repository, as PDF, containing the answers to these questions.


##### Question 1: Markov Decision Processes. 

For this problem, you'll implement the value iteration and policy iteration algorithms. 

I've provided some setup code for you that loads in the states and probabilities, and computes their expected utility.
I've alo provided the setup for two problems - the one shown in R&N (and done in class), and a larger problem, 
the map of which can be found in the file p2.jpg. 
In this second problem, the agent moves in the intended direction with P=0.7, and in each of the other 3 directions with P=0.1.

**Part 1. (1 point)** Get the code working and load in both maps.
 
**Part 2. (9 points)**. Complete computePolicy(). Given utilities set in self.utilities, compute the resulting policy
and return it as a dictionary mapping states to actions.

I've given you a unit test for this.

**Part 3. (15 points)** Implement the value iteration algorithm.

**Part 4. (15 points)** Implement the policy iteration algorithm.

##### Question 2. Reinforcement Learning in Approach.

In this task, we'll return to Approach and use Q-learning to learn the optimal policy through repeated play.

Recall the dice game "Approach". It works like this: There are two players, and they choose a target number (n). Player 1 repeatedly rolls a six-sided die and adds up their total. Whenever they want, they can "hold." Then player 2 must roll;' their goal is to beat player 1's score without going past n.

The problem we want to solve is this:

For player 1, for a given n and a particular total so far (called s) should they 'hold' (action 1) or 'roll' (action 2).

(note that player 2's strategy in this game is pretty boring - they just roll until they either beat player 1's score or exceed n.)

Earlier, we used Monte Carlo simulation to determine the best value to hold. 
This is fine when there's a small number of potential policies to consider, but it's not very scalable. 
Let's now add in Q-learning.

Q-learning generates a table that stores Q(s,a). I've started a function called approach for you. 
It takes in a limit n and computes the Q-table for [0,n] with the actions (hold, roll). 
You will need to complete this.

For example, for n=10, you might get:

<pre>
sum:   hold     roll   [action]
0: 0.000000 0.493921 [roll]
1: 0.000000 0.477271 [roll]
2: 0.000000 0.479286 [roll]
3: 0.000000 0.505797 [roll]
4: 0.000000 0.580803 [roll]
5: 0.051270 0.498027 [roll]
6: 0.170403 0.427388 [roll]
7: 0.297536 0.365115 [roll]
8: 0.478196 0.285432 [hold]
9: 0.711051 0.164235 [hold]
10: 1.000000 0.000000 [hold]
</pre>

We will do this using epsilon-soft on-policy control, a form of reinforcement learning. 
We will select the optimal action with probability (1-epsilon) and explore with probability epsilon. (use epsilon=0.1 for this.)

**(30 points)** Start by initializing the Q-table to small random values. 
Our agent should start with a random s and a and then either take the best action (prob. 1-epsilon), or the other action (prob. epsilon). At the end of the game, it receives reward 1 (if it wins) or 0 (if it loses). 
It then uses this to update the Q value for each state-action pair chosen. Run for 1000000 iterations and print out the Q-table, and optimal action for each state.

### Question 3: Reinforcement Learning for Blackjack

[This tutorial](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/) shows how to apply the Q-learning algorithm to solve 
Blackjack within the Gymnasium environment. The setup is very similar to our previous question, except that epsilon (the probability of selecting a non-optimal action)
decays over time.

**(15 points)** Work through the tutorial and include the code and plots in your repo.

**(15 points)** Write a wrapper program that allows a human player to play against an agent that uses your learned policy. You may either:

a) have the human play directly against the computer. In this case, neither player follows the "hit on 16, hold on 17" hard rule that the dealer follows. The human plays first, and then the computer tries to beat them. Keep track of the number of wins.

b) have both the human and the computer play against a dealer (as would happen in a casino). Keep track of how many wins each player gets.

### Question 4. (Grad students only) Reinforcement Learning with Human Feedback

Reinforcement Learning has proved to be an effective way to fine-tune the question-answering abilities of large language models. Once the models
are initially trained on a corpus, humans are used to then generate feedback that helps the LLM learn to distingush between answers.

[This article from HuggingFace explains more about it](https://huggingface.co/blog/rlhf).

Read this article and answer the following questions:

1. How do humans provide feedback to the LLM?

2. The article describes how to set up an LLM as a reinforcement learning problem. What are the states (or observation space), actions, policy, and reward?

3. Why are there so few RLHF datasets available? 
