
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

    def on_post(self, req, resp):
        dd = json.load(req.stream)
        state = np.array(dd['state'])
        iaction = self.agent.decide_action(state) 
        action = {'action': [int(iaction)]}
        resp.body = json.dumps(action)
        print('-----------CONTENT_TYPE-------------')
        print(req.headers)
        # print('-----------HEADERS-------------')
        # print(req.headers)
        # print('-----------PARAMS-------------')
        # print(req.content_length)
        # print('-----------STREAM-------------')
        # dd = json.load(req.stream)
        # print(dd)


        




        