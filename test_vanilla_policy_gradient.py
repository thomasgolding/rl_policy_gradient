import pytest
import vanilla_policy_gradient as vpg
import numpy as np
import gym



state_dim = 17
n_action = 11
gamma = 0.93
lamb = 0.7
# batchdataparams
n_timesteps = [3,4]



@pytest.fixture
def agent():
    agent = vpg.VanillaPG(state_dim = state_dim, n_action = n_action, gamma = gamma, lamb=lamb)
    return agent

@pytest.fixture
def batchdata():
    batchdata = []
    for n_ep in n_timesteps:
        #state = np.array([np.random.normal(size=(state_dim,)) for _ in range(n_ep)])
        state = np.array([np.ones(shape=(state_dim,)) for _ in range(n_ep)])
        action = np.zeros(n_ep, dtype = 'int')
        reward = np.ones(n_ep)
        batchdata.append({'state': state, 'action': action, 'reward': reward})
    return batchdata

def test_model_shapes(agent):

    batchsize = 13
    s = np.random.normal(size = (batchsize, state_dim))
    val = agent.valuemodel(s).numpy()
    policy = agent.policymodel(s).numpy()
    
    assert val.shape == (batchsize, 1), 'Something wrong with valumodel output-shape'
    assert policy.shape == (batchsize, n_action), 'Something wrong with policymodel output-shape'


def test_sample_action(agent):
    batchsize = 500 
    s = np.random.normal(size = (batchsize, state_dim))
    action = agent.sample_action(s)
    assert action.shape == (batchsize, )
    assert 'int' in str(action.dtype)
    assert action.max() < n_action
    assert action.min() >= 0 


    
def test_sample_batch():
    env = gym.make('CartPole-v0')
    agent = vpg.VanillaPG(state_dim = 4, n_action = 2)
    batchsize = 10
    
    batchdata = vpg.sample_batch(env, agent, batchsize)
    assert len(batchdata) == batchsize
    assert len(batchdata[0]) == 3


def test_process_batchdata(agent, batchdata):
    ## general
    agent.bdat = batchdata
    agent.process_batchdata()
    assert agent.bdat_batchsize == len(n_timesteps)
    assert agent.bdat_ntimesteps == sum(n_timesteps)

    ## reward to go plus test of bdat_slice-data
    batchsize = len(n_timesteps)
    first_rtg = [np.sum(gamma**np.arange(n_ts)) for n_ts in n_timesteps]
    last_rtg = [1.0 for _ in n_timesteps]
    for f_rtg, l_rtg, ind in zip(first_rtg, last_rtg, agent.bdat_slice):
        assert np.abs(agent.bdat_rewardtogo[ind[0]] - f_rtg) < 1.e-5
        assert np.abs(agent.bdat_rewardtogo[ind[1]-1] - l_rtg) < 1.e-5


    ## all states are identical. Check that sum = sum of 1 times number of elements.
    ## this means that there will be noe gaps between trajectories that are unaccoiunted for.
    state = agent.bdat[0]['state'][0,:].reshape((1, state_dim))
    stateval = agent.valuemodel(state).numpy()[0,0]
    expected_sum = agent.bdat_ntimesteps*stateval
    assert np.sum(np.abs(agent.bdat_statevalue)) > 1.e-5
    assert np.abs(np.sum(agent.bdat_statevalue) - expected_sum) < 1.e-5


    
    ## Advantage test
    ## check that the deltas are correct
    for delta, nt in zip(agent.delta, n_timesteps):
        assert delta.shape[0] == nt
        assert np.abs(delta[0] - (1.0 + gamma*stateval - stateval)) < 1.e-5
        assert np.abs(delta[-1] - (1.0 - stateval)) < 1.e-5
        assert np.abs(delta.sum() - nt*(1.0 + gamma*stateval - stateval) + gamma*stateval) < 1.e-5
    


    ## The last advantage of each episode = delta = r - V
    ## r = 1, and V is computed in the model.
    indend = [iend for _, iend in agent.bdat_slice]
    last_advantage = [agent.bdat_advantage[i-1] for i in indend]
    stateval = agent.valuemodel(np.ones((1,state_dim))).numpy()[0,0]
    last_stateval = [stateval]*2
    for l_adv, l_sv in zip(last_advantage, last_stateval):
        assert np.abs(l_adv - (1.0 - l_sv)) < 1.e-5 
    
    ## check the second to last
    factor = lamb*gamma
    stl_advantage = [agent.bdat_advantage[i-2] for i in indend]
    expadv = 1.0+factor + stateval*(gamma-factor-1.0)
    for adv in stl_advantage:
        assert np.abs(adv-expadv) < 1.e-5

    ## check the first advantage of an episode:
    indstart = [istart for istart,_ in agent.bdat_slice]
    assert indstart[0] == 0
    assert indstart[1] == n_timesteps[0]
    first_advantage = [agent.bdat_advantage[i] for i in indstart]
    exp_advantage = []
    for nt in n_timesteps:
        t1 = (1.-stateval)*np.sum(factor**np.arange(nt))
        t2 = gamma*stateval*np.sum(factor**np.arange(nt-1))
        exp_advantage.append(t1 + t2)
    for agent_advantage, expected_advantage in zip(first_advantage, exp_advantage):
        assert np.abs(agent_advantage - expected_advantage) < 1.e-5

    

    ## check logpolicy:
    logits = agent.policymodel(np.ones((1,state_dim))).numpy()
    logits = np.squeeze(logits)
    log_sm = np.log(np.exp(logits)/np.sum(np.exp(logits)))
    exp_logpol = []
    for trajectory in agent.bdat:
        for el in trajectory['action']:
            exp_logpol.append(log_sm[el])
    
    # for logpol, exp_logpol in zip(agent.bdat_logpolicy, exp_logpol):
    #     assert np.abs(logpol - exp_logpol) < 1.e-5
        
         







    


        




