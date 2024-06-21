from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import os

path = os.path.dirname(__file__)

agent = load_prediction_agent()
gtp_go = LocalGtpBot(agent=agent, termination=PassWhenOpponentPasses(),
                        handicap=0, opponent='gnugo', )
gtp_go.run()