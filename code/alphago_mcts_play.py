from dlgo.agent import load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent
from dlgo.httpfrontend.server import get_web_app

fast_policy = load_policy_agent(type='SL', version='v1')
strong_policy = load_policy_agent(type='RL', version='v0101')
value = load_value_agent(version='v1')

alphago = AlphaGoMCTS(strong_policy, fast_policy, value)

web_app = get_web_app({'predict': alphago})

web_app.run()