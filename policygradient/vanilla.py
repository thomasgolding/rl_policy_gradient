import pickle
import numpy as np 
import tensorflow as tf 
import gym
from pathlib import Path

class VanillaPG():
    """
    Vanilla policy gradient
    """
    def __init__(self, 
                 state_dim,
                 n_action, 
                 hidden_policy = [16,16], 
                 hidden_value = [16,16], 
                 gamma = 0.9, 
                 lamb = 0.9,
                 learning_rate_policymodel = 0.01,
                 learning_rate_valuemodel = 0.01,
                 pretrained = False):
        """
        Initfunction
        """

        ## environment
        self.n_action = n_action
        self.state_dim = state_dim

        ## set precision
        self.precision = 'float64'

        ### intervene if loading from file.
        if pretrained:
            filepath = Path(__file__).parent / 'agent_data.pickle'
            if filepath.exists() and not filepath.is_file():
                return
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            hidden_policy = agent_data['hidden_policy']
            hidden_value = agent_data['hidden_value']
            weights_policy = agent_data['policy_weights']
            weights_value = agent_data['value_weights']

            print("Pretrained model loaded.")

        ## policy
        self.neurons_policy = [self.state_dim] + hidden_policy + [self.n_action]
        self.policymodel = self.make_neural_net(self.neurons_policy)
        self.learning_rate_policymodel = learning_rate_policymodel

        ## valuefiunction
        self.neurons_value =  [self.state_dim] + hidden_value + [1]
        self.valuemodel = self.make_neural_net(self.neurons_value, out_activation='linear')
        self.valuemodel.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_valuemodel),
            loss      = tf.keras.losses.MeanSquaredError())

        if pretrained:
            self.policymodel.set_weights(weights_policy)
            self.valuemodel.set_weights(weights_value)

        ## general advantage estimator parameters: 
        self.gamma = gamma    ## discount factor
        self.lamb  = lamb     ## tradeoff bias-variance. lambd = (1,0) means ((low bias high variance), (high bias, low variance))
        
        ## trainparams
        self.batch_number = 0        

    
    def make_neural_net(self, neurons, out_activation = None):
        if len(neurons) < 3: 
            return 
        layers = [tf.keras.layers.InputLayer(input_shape = (neurons[0],), dtype = self.precision)]
        layers += [tf.keras.layers.Dense(units = unit, dtype = self.precision, activation = 'tanh') for unit in neurons[1:-1]]
        layers += [tf.keras.layers.Dense(units = neurons[-1], dtype = self.precision, activation = out_activation)]
        model = tf.keras.models.Sequential(layers)
        return model


    def sample_action(self, state):
        """
        Draw action from policy
        """
        s = state
        if len(state.shape) < 2:
            s = np.expand_dims(state, axis = 0).copy()
        logits = self.policymodel(s)
        action = tf.squeeze(tf.random.categorical(logits = logits, num_samples = 1)).numpy()
        return action


    def decide_action(self, state):
        """ 
        Use this when the agent is controlling actions
        """
        return self.policymodel(state.reshape(1,-1)).numpy().argmax()


    def train_one_batch(self, bdat, verbose = True):
        """
        1. Process batch data:
          - for each episode compute 
             - rewards-to-go: Rhat_t  = sum_i gamma^i*r_{t+i}
             - advantage_t = sum_i (gamma*lambd)(r_t + gamma*V(s_{i+1}) - V(s_t)) 
        2. Compute the policy gradient and update policy
        3. Compute valuefunction-relevant gradient and update valuefunction parameters.
        """

        self.bdat = bdat        
        
        with tf.GradientTape() as tape:
            tape.watch(self.policymodel.trainable_variables)
            self.process_batchdata() ## this provides only nontensors derived
            self.loss = -tf.reduce_sum(self.bdat_logpolicy * tf.Variable(self.bdat_advantage))/self.bdat_batchsize
        #print(self.loss)
        grad = tape.gradient(self.loss, self.policymodel.trainable_variables)
        #print(grad)
        for tv, gr in zip(self.policymodel.trainable_variables, grad):
            tv.assign_sub(self.learning_rate_policymodel * gr) 
        tape.stop_recording()

        ## train model for valuefunction
        fit = self.valuemodel.fit(self.npstate, self.bdat_rewardtogo.reshape(-1,1), epochs=1000, batch_size = 32, verbose=False)
        

        msg = 'batch = {}, trajectories = {}, timesteps = {}, loss_valuefunc = {}'.format(
            self.batch_number,
            self.bdat_batchsize,
            self.bdat_ntimesteps,
            fit.history.get('loss')[-1]
        )

        if verbose:
            print(msg)
             
        self.batch_number += 1

        return msg

    def process_batchdata(self):
        """
        Prepare and orchestrate processing of batch data
        """

        ## get shapes and sizes
        self.bdat_batchsize = len(self.bdat)
        # save also the indexpair defining slices for 
        # each of the episodes in the batch. Use these to access
        # episodes in the flattened reward_to_go, statevalue etc.
        self.bdat_slice = []
        self.bdat_ntimesteps = 0
        istart = 0
        for trajectory in self.bdat:
            ntimestep = len(trajectory['reward'])
            iend = istart + ntimestep
            self.bdat_slice.append((istart, iend))
            istart = iend
            self.bdat_ntimesteps += ntimestep 
        
        self.bdat_rewardtogo = np.zeros(self.bdat_ntimesteps)
        self.bdat_statevalue = np.zeros(self.bdat_ntimesteps)
        self.bdat_advantage = np.zeros(self.bdat_ntimesteps)
        #self.bdat_logpolicy = np.zeros(self.bdat_ntimesteps)

        self.compute_rewardtogo()
        self.compute_statevalue()
        self.compute_advantage() 
        self.compute_logpolicy()


    def compute_rewardtogo(self):
        """
        Compute reward to go on batch.
        """
        iep = 0
        for trajectory in self.bdat:
            nt  = trajectory['reward'].shape[0]
            rtg = trajectory['reward'].copy()
            for ind in reversed(range(nt-1)):
                rtg[ind] += self.gamma*rtg[ind+1]
            self.bdat_rewardtogo[iep:iep+nt] = rtg
            iep += nt


    def compute_statevalue(self):
        ## compute valuefunc
        iep = 0
        for trajectory in self.bdat:
            stateval = np.squeeze(self.valuemodel(trajectory['state']).numpy())
            nt = stateval.shape[0]
            self.bdat_statevalue[iep:iep+nt] = stateval
            iep += nt


    def compute_advantage(self):
        # compute the deltas
        factor = self.gamma*self.lamb
        self.delta = []
        for trajectory, iep in zip(self.bdat, range(self.bdat_batchsize)):
            stateval = self.get_bdat_estimate_from_episode(iep = iep, var = 'statevalue')
            tmp_delta = trajectory['reward'] - stateval
            tmp_delta[0:-1] += self.gamma*stateval[1:]
            self.delta.append(tmp_delta)

            adv = tmp_delta.copy()
            nn = len(stateval)
            for it in reversed(range(nn-1)):
                adv[it] += factor*adv[it+1]
            
            istart, iend = self.bdat_slice[iep]
            self.bdat_advantage[istart:iend] = adv.copy()


    def compute_logpolicy(self):
        npstate = np.zeros((self.bdat_ntimesteps, self.state_dim))
        npaction = np.zeros((self.bdat_ntimesteps))
        for trajectory, (istart, iend) in zip(self.bdat, self.bdat_slice):
            npstate[istart:iend,:] = trajectory['state']
            npaction[istart:iend] = trajectory['action']
        ## debug
        self.npstate = npstate.copy()
        
        self.state_tensor = tf.Variable(npstate)
        self.action_tensor = tf.Variable(npaction, dtype = 'int32')

        self.tf_logpolicy()

    #@tf.function
    def tf_logpolicy(self):
        logits = self.policymodel(self.state_tensor)
        log_softmax = tf.nn.log_softmax(logits)
        mask = tf.one_hot(self.action_tensor, logits.shape[1], dtype = 'float64')
        self.bdat_logpolicy = tf.reduce_sum(log_softmax*mask, axis=1)


    def get_bdat_estimate_from_episode(self, iep, var):
        (istart, iend) = self.bdat_slice[iep]
        res = 0
        if (var == 'rewardtogo'):
            res = self.bdat_rewardtogo[istart:iend].copy()
        elif (var == 'statevalue'):
            res = self.bdat_statevalue[istart:iend].copy()
        elif (var == 'advantage'):
            res = self.bdat_advantage[istart:iend].copy()
        elif (var == 'log_policy'):
            res = self.bdat_logpolicy[istart:iend].copy()
        return res
            
            
    def save_agent(self, filename='agent_data.pickle'):
        agent_data = {}
        agent_data['policy_weights'] = self.policymodel.get_weights()
        agent_data['value_weights'] = self.valuemodel.get_weights()
        agent_data['hidden_policy'] = self.neurons_policy[1:-1]
        agent_data['hidden_value'] = self.neurons_value[1:-1]
        agent_data['state_dim'] = self.state_dim
        agent_data['n_action'] = self.n_action
        with open(filename, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print('saved agent to {}'.format(filename))
        
 

def sample_batch(env, agent, batchsize):
    """
    Sample batch with actions drawn from agent's policy
    
    One trajectory: (s0, a0, r0, s1, a1, r1, .... sn, an, rn, s_n+1)
    sample batsize number of trajectories and return a list of batchsize  
    trajectory-dictionaries. Not including sn+1.
    
    trajecory = {
        'state'  = [s0, s1, s2, .... sn],
        'action' = [a1, a2, a3, ..., an],
        'reward' = [r1, r2, r3, ..., rn]
    }
    """
    batchdata = []

    for iep in range(batchsize):    
        sl = []
        rl = []
        al = []
        s = env.reset()
        done = False
        while not done:
            a = agent.sample_action(s)
            [s2, r, done, _] = env.step(a)
            if done: r = -10. 
            sl.append(s)
            rl.append(r)
            al.append(a)
            s = s2.copy()
        
        trajectory = {'state': np.array(sl), 'action': np.array(al), 'reward': np.array(rl)}
        batchdata.append(trajectory)
    
    return batchdata


def n_timesteps(a, envir):
    s = envir.reset()
    for it in range(510):
        s = np.expand_dims(s, axis = 0)
        action = a.decide_action(s)
        [s,_,d,_] = envir.step(action)
        if d:
            break
    return it


def exp_timestep(a, envir):
    n = 50
    nt = np.zeros(n)
    for i in range(n):
        nt[i] = n_timesteps(a, envir)
    return  nt.mean()



### train and save model.
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = VanillaPG(state_dim=4, n_action = 2, gamma=0.9, lamb = 0.5, learning_rate_policymodel=0.003)
    for _ in range(50):
        aa = sample_batch(env, agent, 32)
        msg = agent.train_one_batch(aa, verbose = False)
        exp_nt = exp_timestep(agent, env)
        msg = 'exp_nt = {}, {}'.format(exp_nt, msg)
        print(msg)
        if exp_nt > 250:
            break
    agent.save_agent()
    

#    print(agent.bdat_statevalue.reshape(-1,1).shape)
    #print('advantage')
    #print(agent.bdat_advantage)
    #print('logpolict')
   #print(agent.bdat_logpolicy)
    #print('sum_elems')
    #print(agent.loss)
    #agent.process_batchdata()
    #print(aa[0]['action'])
    #print(np)
    #print(agent.action_tensor)
    #print(agent.policymodel(aa[0]['state']))
    #state = np.expand_dims(np.array([10.,20.,-300.,4.]), 0)
    #print(agent.valuemodel(np.ones((1,4))))
    #state = np.random.normal(size = (1,4))
    #print(agent.sample_action(state))

