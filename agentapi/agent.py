
import json
import falcon
import policygradient
import numpy as np


class Agent():
    def __init__(self):
        self.agent = policygradient.VanillaPG(state_dim = 4, n_action = 2, pretrained = True)
    
    def on_get(self, req, resp):
        state = [float(el) for el in req.params['state']]
        state = np.array(state)
        print(state)
        iaction = self.agent.decide_action(state) 
        action = {'action': [int(iaction)]}
        resp.body = json.dumps(action)
        resp.status = falcon.HTTP_200