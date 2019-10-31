import requests

host = 'http://127.0.0.1'
port = '8000'
endpoint = '/agent'

url = host + ':' + port + endpoint

state = {'state': [0.01,0.02,0.01,0.01]}

res = requests.get(url, params = state)

print(res.json())

