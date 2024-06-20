from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import h5py
import os

path = os.path.dirname(__file__)

agent = load_prediction_agent()
gnu_go = LocalGtpBot(agent=agent, termination=PassWhenOpponentPasses(),
                        handicap=0, opponent='pachi', )
gnu_go.run()