import argparse

from dlgo import agent
from dlgo import rl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience', nargs='+')

    args = parser.parse_args()
    learning_agent_filename = args.learning_agent
    experience_files = args.experience
    updated_agent_filename = args.agent_out
    learning_rate = args.lr
    clipnorm = args.clipnorm
    batch_size = args.bs

    learning_agent = agent.load_policy_agent(name=learning_agent_filename)
    for exp_filename in experience_files:
        winning_exp_buffer, losing_exp_buffer = rl.load_experience(exp_filename)
        learning_agent.train(
            winning_exp_buffer,
            losing_exp_buffer,
            lr=learning_rate,
            clipnorm=clipnorm,
            batch_size=batch_size
        )
    
    learning_agent.serialize(updated_agent_filename)


if __name__ == '__main__':
    main()