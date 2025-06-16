import constants
from montecarlo import MonteCarloAgent
import pickle

if __name__ == "__main__":
    agent = MonteCarloAgent()
    agent.run(
        num_episodes=constants.NUM_OF_EPISODES,
        layout="rectangle",
        rows=3,
        cols=3
    )
    with open("pickle/mc_agent_rectangle.pkl", "wb") as f:
        pickle.dump(agent, f)
