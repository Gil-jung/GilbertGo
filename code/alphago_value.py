from dlgo.networks.alphago import AlphaGoValueNet
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl import ValueAgent, load_experience

rows, cols = 19, 19
encoder = AlphaGoEncoder(use_player_plane=True)
model = AlphaGoValueNet()

alphago_value = ValueAgent(model, encoder)

winning_exp_buffer, losing_exp_buffer = load_experience(name='0001')

alphago_value.train(
    winning_exp_buffer, losing_exp_buffer,

)

alphago_value.serialize(version='v0')