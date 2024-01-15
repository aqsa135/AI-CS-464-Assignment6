from random import randint
from random import random

# Approach.py

# This program uses reinforcement learning to determine the optimal policy
# for Approach.
# Recall that approach works like this:
# Both players agree on a limit n.
# Player 1 rolls first. They go until they either exceed n or hold.
# Then player 2 rolls. They go until they either exceed n or beat player 1's score.
# The player who is closest to n without going over wins.
# Note:
# We can reduce this to the problem of player 1 choosing the best value at which to hold.
# This is called a policy; once we know the best number to hold at, we can act optimally.


#     for i in range(100000) :
        # Select an initial state.
        # Take the best move with p=epsilon, and the worst move with p=1-epsilon.
        # Continue playing until the game is done.
        # If you win, reward = 1.
        # If you lose, reward = 0.
        ## Use Q-learning to update the q-table for each state-action pair visited.

    ## After 100000 iterations, print out your q-table.


def approach(n, epsilon=0.1, alpha=0.1, gamma=0.9):
    q_table = [[random() / 100.0, random() / 100.0] for _ in range(n + 1)]

    for _ in range(1000000):
        s = randint(0, n-1)  # Start from a random state
        done = False

        while not done:
            if random() < epsilon:
                # Exploration: choose a random action
                action = randint(0, 1)
            else:
                # Exploitation: choose the best action from Q-table
                action = q_table[s].index(max(q_table[s]))

            # Simulate the game to get new state and reward
            if action == 0:  # Hold
                player_2_score = 0
                while player_2_score <= s:
                    player_2_score += randint(1, 6)
                    if player_2_score > n:
                        break
                reward = 1 if player_2_score > n else 0
                done = True
                new_s = 0  # Reset the state
            else:  # Roll
                roll = randint(1, 6)
                new_s = s + roll
                if new_s > n:
                    reward = 0
                    done = True
                else:
                    reward = 0

            # Update Q-table
            old_value = q_table[s][action]
            if new_s <= n:
                next_max = max(q_table[new_s])
            else:
                next_max = 0  # No future reward if the new state is beyond n

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[s][action] = new_value

            s = new_s  # Move to the next state

    # Print the Q-table
    for s in range(n + 1):
        print(f'{s}: {q_table[s]} [hold]' if q_table[s][0] > q_table[s][1] else f'{s}: {q_table[s]} [roll]')

approach(10)


# Q-table is initialized with small random values for each state-action pair
# epsilon-soft policy for action selection. The agent chooses the best action (action with the highest Q-value) with probability
# 1−ϵ and a random action with probability ϵ.
#Player 1 (represented by the agent) either holds or rolls based on the selected action.
# If the player holds, Player 2's turn is simulated, and the game outcome is determined.
# If the player rolls, the score is updated and checked against the limit n.
# reward is set to 1 if Player 1 wins (i.e., Player 2 exceeds n after Player 1 holds) and 0 otherwise
#  the Q-values for the state-action pairs visited during the game are updated based on the reward and the estimated future rewards, after each game
#^ above is done using the learning rate alpha and the discount factor gamma.
# to allow the Q-values to converge towards optimal values, the process is repeated for 1,000,000 iterations
# the Q-table is printed, showing the Q-values for each state-action pair and the optimal action for each state i.e., roll or hold, after the iterations
# approach(10) will run algorithm for n= 10 and print resulting Q-table