from collections import defaultdict
import random
## transitions
## The transition probabilities are stored in a dictionary mapping (state, action) pairs to a list
## of edges - (tuples indicating destinations and probabilities)

class MDP :

    def __init__(self,gamma=0.8, error=0.01, mapfile=None, reward=-0.04):
        self.gamma=gamma
        self.error=error
        self.reward=reward
        if mapfile :
            self.goals, self.transition_probs = load_map_from_file(mapfile)
        else :
            self.goals = []
            self.transition_probs=defaultdict(list)
        self.states =  set([item[0] for item in self.transition_probs.keys()] + [item[0] for item in self.goals])
        self.actions = set([item[1] for item in self.transition_probs.keys()])
        self.utilities = defaultdict(float)
        for item in self.goals :
            self.utilities[item[0]] = float(item[1])
        for item in self.states :
            self.utilities[item] = 0.1

    def __repr__(self):
        return f"Gamma: {self.gamma}, Error: {self.error}, Reward: {self.reward}, Goals: {self.goals}, Transitions: {self.transition_probs}, States: {self.states}, Actions: {self.actions}"

    ## return the policy, represented as a dictionary mapping states to actions, for the current utilities.
    ## you do this.
    def computePolicy(self):
        policy = {}
        for state in self.states:
            if state in self.goals:  # Skip goal states
                continue

            best_action = None
            max_expected_utility = float('-inf')

            for action in self.actions:
                expected_utility = 0
                for (probability, next_state) in self.transition_probs[(state, action)]:
                    expected_utility += float(probability) * self.utilities[next_state]

                if expected_utility > max_expected_utility:
                    max_expected_utility = expected_utility
                    best_action = action

            policy[state] = best_action

        return policy

    ## for a state, compute its expected utility
    def computeEU(self, state):
        ## are we at a goal?
        for goal in self.goals :
            if state == goal[0] :
                return goal[1]
        ## if not, for each possible action, get all the destinations and compute their EU. keep the max.
        best_action = None
        best_eu = -1.0
        for action in self.actions :
            eu = 0.0
            destinations = self.transition_probs[(state, action)]
            for d in destinations :
                eu += self.utilities[d[1]] * float(d[0])
            if eu >= best_eu :
                best_action = action
                best_eu = eu
        return best_eu

    ## you do this one.
    ## 1. Initialize the utilities to random values.
    ## 2 do:
    ##     for state in states:
    ##           compute its new EU
    ##     update all values
    ##  while any EU changes by more than delta = (1-error)/error
    ##

    def valueIteration(self, error=0.01):
        """
        Perform the value iteration algorithm.
        """
        # 1. Initialize the utilities to random values (excluding goal states)
        self.utilities = {state: random.random() for state in self.states
                          if state not in [goal[0] for goal in self.goals]}

        # Set utility for goal states to their defined values
        for goal, utility in self.goals:
            self.utilities[goal] = float(utility)

        delta = (1 - error) / error  # Convergence threshold

        while True:
            max_utility_change = 0
            new_utilities = self.utilities.copy()

            # 2. Compute new expected utility for each state
            for state in self.states:
                if state in [goal[0] for goal in self.goals]:  # Skip goal states
                    continue

                max_utility = float('-inf')
                for action in self.actions:
                    total = sum(float(prob) * self.utilities[next_state]
                                for prob, next_state in self.transition_probs[(state, action)])
                    if total > max_utility:
                        max_utility = total

                new_utilities[state] = self.reward + self.gamma * max_utility
                max_utility_change = max(max_utility_change, abs(new_utilities[state] - self.utilities[state]))

            # Update the utilities
            self.utilities = new_utilities

            # Check for convergence
            if max_utility_change < delta:
                break

    ## you do this one.
    ## 1. Set all utilities to zero.
    ## 2. Generate a random policy.
    ## do :
    ##    given the policy, update the utilities.
    ##    call computePolicy to get the policy for these utilities.
    ## while: any part of the policy changes.

    def policyIteration(self):
        """
        Perform the policy iteration algorithm.
        """
        # 1. Set all utilities to zero.
        self.utilities = {state: 0 for state in self.states}

        # 2. Generate a random policy.
        policy = {state: None if state in [goal[0] for goal in self.goals] else random.choice(list(self.actions))
                  for state in self.states}

        while True:
            # Update utilities given the current policy
            while True:
                max_utility_change = 0
                new_utilities = self.utilities.copy()
                for state in self.states:
                    if policy[state] is None:  # Skip goal states
                        continue

                    total = sum(float(prob) * self.utilities[next_state]
                                for prob, next_state in self.transition_probs[(state, policy[state])])

                    new_utilities[state] = self.reward + self.gamma * total
                    max_utility_change = max(max_utility_change, abs(new_utilities[state] - self.utilities[state]))

                self.utilities = new_utilities

                if max_utility_change < 0.01:
                    break

            # Compute the new policy based on updated utilities
            new_policy = self.computePolicy()

            # Check if any part of the policy has changed
            policy_changed = any(new_policy[state] != policy[state] for state in self.states)

            if not policy_changed:
                break

            policy = new_policy

        return policy



def load_map_from_file(fname) :
    goals = []
    transitions = defaultdict(list)
    with open(fname) as f:
        for line in f :
            if line.startswith("#") or len(line) < 2 :
                continue
            elif line.startswith('goals') :
                goals = [tuple(x.split(':')) for x in line.split()[1:]]
            else :
                source, action, dests = line.split(' ', 2)
                transitions[(source, action)]=[tuple(x.split(':')) for x in dests.split()]
    return goals, transitions