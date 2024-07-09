import argparse

from dlgo.elo import calculate_ratings
from dlgo.agent.pg import load_policy_agent
from dlgo.rl.value import load_value_agent
from dlgo.agent.alphago import AlphaGoMCTS
from dlgo.zero.agent import load_zero_agent
from dlgo.mcts import MCTSAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-g', type=int)
    parser.add_argument('--board-size', '-b', type=int)
    parser.add_argument('agents', nargs='+')

    args = parser.parse_args()

    agent1 = load_policy_agent(type='SL', version='v6')
    agent2 = load_policy_agent(type='RL', version='v0')
    agent3 = load_value_agent(version='v0')
    agent4 = AlphaGoMCTS(
        policy_agent=agent2, fast_policy_agent=agent1, value_agent=agent3, 
        lambda_value=0.5, num_simulations=10, depth=200, rollout_limit=10
    )
    agent5 = load_zero_agent(version='v0')
    agent6 = MCTSAgent(num_rounds=500, temperature=1.4)
    agents = [
        agent1, agent2, agent3, agent4, agent5, agent6,
    ]

    # for a in agents:
    #     a.set_temperature(0.02)

    ratings = calculate_ratings(agents, args.num_games, args.board_size)

    for filename, rating in zip(args.agents, ratings):
        print("%s %d" % (filename, rating))


if __name__ == '__main__':
    main()