from montecarlo import MonteCarloAgent
import constants
import pickle

if __name__ == "__main__":
    agent = MonteCarloAgent()
    agent.run(num_episodes=1000, layout=constants.BRICK_LAYOUT, rows=3, cols=constants.BRICK_COLUMNS)
    # Save agent for reuse
    with open("mc_agent.pkl", "wb") as f:
        pickle.dump(agent, f)
