import vanilla as vpg
import gym 

agent = vpg.VanillaPG(state_dim = 4, n_action = 2, pretrained=True)
env = gym.make('CartPole-v1')

print(vpg.exp_timestep(agent, env))