import falcon
from .agent import Agent

api = application = falcon.API()

agent = Agent()
api.add_route('/agent', agent)

