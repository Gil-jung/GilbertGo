from dlgo.agent.naive import RandomBot
from dlgo.minimax.depthprune import DepthPrunedAgent 
from dlgo.minimax.alphabeta import AlphaBetaAgent
from dlgo.mcts import MCTSAgent
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.agent.pg import load_policy_agent
from dlgo.httpfrontend.server import get_web_app

# agent = load_policy_agent()
agent = load_prediction_agent()
# agent = MCTSAgent()
# agent = AlphaBetaAgent()
# agent = RandomBot()
web_app = get_web_app({'predict': agent})
# web_app = get_web_app({'random': agent})
web_app.run()