from dlgo.agent.pg import load_policy_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks.alphago import AlphaGoValueResNet
from dlgo.rl.simulate import experience_simulation
from dlgo.rl.value import ValueAgent
from dlgo.rl.experience import ExperienceBuffer

import numpy as np


def main():
    alphago_rl_agent = load_policy_agent(type='SL', version='v0')
    opponent = load_policy_agent(type='SL', version='v0')
    alphago_rl_agent._model.cuda()
    opponent._model.cuda()

    encoder = AlphaGoEncoder(use_player_plane=True)
    model = AlphaGoValueResNet()
    value_agent1 = ValueAgent(model, encoder)
    value_agent2 = ValueAgent(model, encoder)

    num_games = 10
    policy_exp_buffer, value_exp_buffer = experience_simulation(
        num_games, alphago_rl_agent, opponent, value_agent1, value_agent2
    )
    
    winning_policy_states = policy_exp_buffer[0].states
    winning_policy_actions = policy_exp_buffer[0].actions
    losing_policy_states = policy_exp_buffer[1].states
    losing_policy_actions = policy_exp_buffer[1].actions
    winning_value_states = value_exp_buffer[0].states
    winning_value_rewards = value_exp_buffer[0].rewards
    losing_value_states = value_exp_buffer[1].states
    losing_value_rewards = value_exp_buffer[1].rewards
    
    chunk = 0
    chunk_size = 1024
    while len(winning_policy_actions) >= chunk_size:
        current_winning_policy_states, winning_policy_states = winning_policy_states[:chunk_size], winning_policy_states[chunk_size:]
        current_winning_policy_actions, winning_policy_actions = winning_policy_actions[:chunk_size], winning_policy_actions[chunk_size:]
        current_losing_policy_states, losing_policy_states = losing_policy_states[:chunk_size], losing_policy_states[chunk_size:]
        current_losing_policy_actions, losing_policy_actions = losing_policy_actions[:chunk_size], losing_policy_actions[chunk_size:]
        current_winning_value_states, winning_value_states = winning_value_states[:chunk_size], winning_value_states[chunk_size:]
        current_winning_value_rewards, winning_value_rewards = winning_value_rewards[:chunk_size], winning_value_rewards[chunk_size:]
        current_losing_value_states, losing_value_states = losing_value_states[:chunk_size], losing_value_states[chunk_size:]
        current_losing_value_rewards, losing_value_rewards = losing_value_rewards[:chunk_size], losing_value_rewards[chunk_size:]

        ExperienceBuffer(
            current_winning_policy_states,
            current_winning_policy_actions,
            [],
            []
        ).serialize(result="winning", name=f'policy_{chunk}')
        ExperienceBuffer(
            current_losing_policy_states,
            current_losing_policy_actions,
            [],
            []
        ).serialize(result="losing", name=f'policy_{chunk}')
        ExperienceBuffer(
            current_winning_value_states,
            [],
            current_winning_value_rewards,
            []
        ).serialize(result="winning", name=f'value_{chunk}')
        ExperienceBuffer(
            current_losing_value_states,
            [],
            current_losing_value_rewards,
            []
        ).serialize(result="losing", name=f'value_{chunk}')

        chunk += 1
    
    ExperienceBuffer(
        winning_policy_states,
        winning_policy_actions,
        [],
        []
    ).serialize(result="winning", name=f'policy_{chunk}')
    ExperienceBuffer(
        losing_policy_states,
        losing_policy_actions,
        [],
        []
    ).serialize(result="losing", name=f'policy_{chunk}')
    ExperienceBuffer(
        winning_value_states,
        [],
        winning_value_rewards,
        []
    ).serialize(result="winning", name=f'value_{chunk}')
    ExperienceBuffer(
        losing_value_states,
        [],
        losing_value_rewards,
        []
    ).serialize(result="losing", name=f'value_{chunk}')

    alphago_rl_agent.train(
        lr=0.001,
        clipnorm=1.0,
        batch_size=128
    )

    alphago_rl_agent.serialize(type='RL', version='v0')


if __name__ == '__main__':
    main()