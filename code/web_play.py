from dlgo.agent.naive import RandomBot
from dlgo.minimax.depthprune import DepthPrunedAgent 
from dlgo.minimax.alphabeta import AlphaBetaAgent
from dlgo.httpfrontend.server import get_web_app


agent = AlphaBetaAgent()
# agent = RandomBot()
web_app = get_web_app({'random': agent})
web_app.run()