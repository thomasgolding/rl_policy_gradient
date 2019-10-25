import vanilla_policy_gradient as vpg
import gym 

agent = vpg.VanillaPG(state_dim = 4, n_action = 2, load_agent_file='agent_data.pickle')
env = gym.make('CartPole-v1')

print(vpg.exp_timestep(agent, env))