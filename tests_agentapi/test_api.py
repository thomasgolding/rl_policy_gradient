import falcon

from falcon import testing
import json
import numpy as np
import pytest

from agentapi.api import api

@pytest.fixture
def client():
    return testing.TestClient(api)

def test_decide_action(client):
    data = json.dumps({'state': [0.01,0.02,0.01,0.01]})
    #bytecodedata = data.encode('UTF-8')
    headers = {'content-type': 'application/json'}

    response = client.simulate_post('/agent', body = data, headers = headers)
    d = response.json
    assert isinstance(d['action'], list)
    assert isinstance(d['action'][0], int) 
    


