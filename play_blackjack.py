from blackjack import BlackjackAgent
import gymnasium as gym
from tqdm import tqdm
# Training function for the BlackjackAgent


# part b of question 3 DONE (I did b)
def train_agent(agent, env, n_episodes=100000):
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

# Function for playing Blackjack: human vs. computer
def human_vs_computer(agent, env):
    human_wins = 0
    computer_wins = 0
    games_to_play = 10

    for _ in range(games_to_play):
        observation, info = env.reset()

        # Human's turn
        while True:
            print(f"Your hand: {observation}")
            action = int(input("Enter your action (0: Stand, 1: Hit): "))
            observation, reward, done, _, _ = env.step(action)
            if done:
                if reward > 0:
                    human_wins += 1
                break

        # Computer's turn
        if not done:
            while True:
                action = agent.get_action(observation)
                observation, reward, done, _, _ = env.step(action)
                if done:
                    if reward > 0:
                        computer_wins += 1
                    break

        print(f"Game result: Human {human_wins}, Computer {computer_wins}")

    return human_wins, computer_wins

# Main 
if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    agent = BlackjackAgent(
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=1.0 / (100000 / 2),
        final_epsilon=0.1,
    )

    # Train the agent
    print("Training the agent...")
    train_agent(agent, env)

    # Play the game
    print("\nLet's play Blackjack!")
    human_wins, computer_wins = human_vs_computer(agent, env)
    print(f"Final Score - Human: {human_wins}, Computer: {computer_wins}")