from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl.simulate import experience_simulation

encoder = AlphaGoEncoder()

sl_agent = load_prediction_agent()
sl_opponent = load_prediction_agent()

alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
opponent = PolicyAgent(sl_opponent.model, encoder)

num_games = 1000
winning_experiences, losing_experiences = experience_simulation(num_games, alphago_rl_agent, opponent)

alphago_rl_agent.train(winning_experiences, losing_experiences)

alphago_rl_agent.serialize(version='v0')
experience.serialize()