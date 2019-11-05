
import json
import falcon
import policygradient
import numpy as np


class Agent():
    def __init__(self):
        self.agent = policygradient.VanillaPG(state_dim = 4, n_action = 2, pretrained = True)
    
    def on_get(self, req, resp):
        state = np.array([0.1,0.1,0.1,0.1])
        iaction = self.agent.decide_action(state) 
        action = {'action': [int(iaction)]}
        resp.body = json.dumps(action)
        resp.status = falcon.HTTP_200 if int(iaction) in [0,1] else falcon.HTTP_404

    def on_post(self, req, resp):
        dd = req.stream.read(req.content_length)
        dd = json.loads(dd)
        state = np.array(dd['state'])
        iaction = self.agent.decide_action(state) 
        action = {'action': [int(iaction)]}
        resp.body = json.dumps(action)


        # print('-----------CONTENT_TYPE-------------')
        # print(req.headers)
        # print('-----------HEADERS-------------')
        # print(req.headers)
        # print('-----------PARAMS-------------')
        # print(req.content_length)
        # print('-----------STREAM-------------')
        #print(type(req.stream.read()))
        #dd = json.load(req.stream)
        #print(dd)


        




        