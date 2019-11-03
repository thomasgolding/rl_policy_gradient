import gym
import requests
import json

def run_cartpole_on_agentapi():
    env = gym.make('CartPole-v1')
    s = env.reset()
    done = False
    env.render()
    while not done:
        d = json.dumps({'state': s.tolist()})
        resp = requests.post('http://127.0.0.1:5000/agent', data=d)
        a = resp.json()['action'][0]
        [s2, r, done, _] = env.step(a)
        s = s2.copy()
        env.render()


if __name__ == '__main__':
    run_cartpole_on_agentapi()
    





