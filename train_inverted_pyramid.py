import constants
from montecarlo import MonteCarloAgent
import pickle

if __name__ == "__main__":
    agent = MonteCarloAgent()
    agent.run(
        num_episodes=constants.NUM_OF_EPISODES,
        layout="inverted_pyramid",
        rows=3,
        cols=3,
        ball_start_direction=0
    )
    with open("pickle/mc_agent_inverted_pyramid.pkl", "wb") as f:
        pickle.dump(agent, f)
