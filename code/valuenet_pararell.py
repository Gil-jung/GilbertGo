from dlgo.networks.alphago import AlphaGoValueResNet, AlphaGoValueMiniResNet
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.rl.value import ValueAgent
from dlgo.rl.simulate import value_experience_simulation
from dlgo.rl.experience import ExperienceBuffer
import os, sys, multiprocessing
import torch
import torch.nn as nn

chunk = 0

def exp_file_generation(value_exp_buffer):
    global chunk
    winning_value_states = value_exp_buffer[0].states
    winning_value_rewards = value_exp_buffer[0].rewards
    losing_value_states = value_exp_buffer[1].states
    losing_value_rewards = value_exp_buffer[1].rewards
    
    # chunk = 0
    chunk_size = 1024
    while len(winning_value_rewards) >= chunk_size:
        current_winning_value_states, winning_value_states = winning_value_states[:chunk_size], winning_value_states[chunk_size:]
        current_winning_value_rewards, winning_value_rewards = winning_value_rewards[:chunk_size], winning_value_rewards[chunk_size:]
        current_losing_value_states, losing_value_states = losing_value_states[:chunk_size], losing_value_states[chunk_size:]
        current_losing_value_rewards, losing_value_rewards = losing_value_rewards[:chunk_size], losing_value_rewards[chunk_size:]

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


def main():
    self_play_mode = True
    training_mode = False
    if self_play_mode:
        cores = multiprocessing.cpu_count()
        value_exp_buffers = []
        for x in range(cores):
            value_exp_buffers.append(value_experience_simulation(
            num_games=100, agent1='m2', agent2=['m2']
        ))
        # ray.init(num_cpus=6, runtime_env={"working_dir": os.path.dirname(__file__)})

        # exp_file_generation(value_exp_buffer)
        # value_exp_buffer = ray.put(value_exp_buffer)
        # task = exp_file_generation.remote(value_exp_buffer)
        # ray.get([exp_file_generation.remote(value_exp_buffer) for i in range(6)])
        # ray.shutdown()
        with multiprocessing.Pool(processes=cores) as p:
            result = p.map_async(exp_file_generation, value_exp_buffers)
            try:
                _ = result.wait()
            except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
                p.terminate()
                p.join()
                sys.exit(-1)


    if training_mode:
        path = os.path.dirname(__file__)
        pre_trained = False
        version = 0
        # encoder = AlphaGoEncoder(use_player_plane=True)      
        encoder = SimpleEncoder(board_size=(19, 19))
        # model = AlphaGoValueResNet()          
        model = AlphaGoValueMiniResNet(num_planes=11)
        
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

        if not pre_trained:
            model.apply(initialize_weights)
            print("initializing...")
        else:
            pt_flie = torch.load(path + f"\\agents\\AlphaGo_Value_Agent_m{version}.pt")
            print("model loading...")
            model.load_state_dict(pt_flie['model_state_dict'])

        alphago_value = ValueAgent(model, encoder)

        alphago_value.train(
            lr=0.001,
            batch_size=128,
        )

        # alphago_value.serialize(version='v0')


if __name__ == '__main__':
    main()