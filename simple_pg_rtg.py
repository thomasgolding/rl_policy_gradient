##  Reward to go only. Only let th

import tensorflow as tf 
import numpy as np 

import gym 

game = 'CartPole-v1'

def make_policymodel(state_dim, n_action, hidden_size = [32], activation = 'relu', out_activation = None):
    layers = [tf.keras.layers.InputLayer(input_shape = (state_dim,),dtype='float64')]
    layers += [tf.keras.layers.Dense(n_unit, activation=activation, dtype = tf.float64) for n_unit in hidden_size]
    layers += [tf.keras.layers.Dense(n_action, activation = out_activation, dtype = tf.float64)]
    policymodel = tf.keras.models.Sequential(layers)
    return policymodel

#@tf.function
def compute_logits(mod, state):
    return mod(state)

#@tf.function
def sample_action(mod, state):

    logits = compute_logits(mod, state)
    act = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1))
    return act

# @tf.function
# def lossfunc(model, batch_rew, batch_action, batch_states, n_action):
#     logits = model(batch_states)
#     action_mask = tf.one_hot(batch_action, n_action)
#     loss = tf.reduce_sum(action_mask)

#     return loss #tf.reduce_sum(batch_rew)

#@tf.function
def compute_loss(model, states, actions, rewards, batchsize):
    logits = compute_logits(model, states)
    action_mask = tf.one_hot(actions, logits.shape[1], dtype = 'float64')
    weights = rewards# tf.expand_dims(rewards, axis=1)    
    
    tmp = tf.reduce_sum(action_mask*tf.nn.log_softmax(logits), axis=1)*weights
    
    res = - tf.reduce_sum(tmp)/batchsize

    #res = tmp #tf.nn.softmax(logits)
    return res

def play_batch(batchsize, env, mod):
    l_reward  = []
    l_state  = []
    l_action = []

    for i_ep in range(batchsize):
        s = env.reset()
        done = False
        #total_reward = 0
        ts_in_ep = 0
        l_rew_ep = []
        while not done:
            reshaped_s = np.expand_dims(s, 0)
            action = sample_action(mod, reshaped_s).numpy()
            [newstate, reward, done, _] = env.step(action)
            
            l_action.append(action)
            l_state.append(s)
            #total_reward += reward
            l_rew_ep.append(reward)
            ts_in_ep += 114
            s = newstate.copy()

        tmp = np.cumsum(l_rew_ep)
        rew_to_go = list(tmp.max() - tmp)
        l_reward += rew_to_go
    
    batchdata = { 
        "rewards": np.array(l_reward),
        "states":  np.array(l_state),
        "actions": np.array(l_action),
        "batchsize": batchsize
    }
    
    return batchdata





def train(n_batch, batchsize):
    env = gym.make(game)
    n_action = env.action_space.n 
    state_dim = env.observation_space.shape[0]
    policymodel = make_policymodel(state_dim=state_dim, n_action=n_action, hidden_size=[32])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)


    for ibatch in range(n_batch):
        batchdata = play_batch(batchsize = batchsize, env = env, mod = policymodel)
        states = tf.Variable(batchdata['states'])
        actions = tf.Variable(batchdata['actions'])
        rewards = tf.Variable(batchdata['rewards'])
        xbatchsize = tf.Variable(batchdata['batchsize'], dtype = 'float64')
        with tf.GradientTape() as tape:
            loss = compute_loss(model = policymodel, states = states, 
                                actions = actions, rewards = rewards,
                                batchsize = xbatchsize)        
            # loss = compute_loss(model = policymodel, states = batchdata['states'], 
            #                     actions = batchdata['actions'], rewards = batchdata['rewards'],
            #                     batchsize = batchdata['batchsize'])        
        grad = tape.gradient(loss, policymodel.trainable_variables)
        #for g, v in zip(grad, policymodel.trainable_variables):
        #    v.assign_sub(0.01*g)
        optimizer.apply_gradients(zip(grad, policymodel.trainable_variables))

        eval_data = play_batch(batchsize=1, env=env, mod = policymodel)
        ntimestep = eval_data['rewards'][0]
                
        print("ibatch = {}   ntimestep = {}".format(ibatch, ntimestep))

    res = policymodel
    return res


if __name__ == '__main__':
    model = make_policymodel(state_dim = 4, n_action = 2)
    env = gym.make('CartPole-v0')
    s = env.reset().reshape((1,4)).astype('float64')
    
    s3 = np.array(4*[env.reset()])
    #print(s3)
    actions = np.array([0,1,1,0])
    rewards = np.array([1,1,2,2], dtype = 'float64')
    #res = compute_loss(model, s3, actions,rewards)
    # print(res)

    # bdat = play_batch(1, env, model)
    # print(bdat)

    grad = train(n_batch=20, batchsize=500)
    print(grad)




#     def train_one_batch(n_episode):
#         # loop through
#         batch_rewards = []
#         batch_states = []
#         batch_actions = []
#         for iep in range(n_episode):
#             s = env.reset()
#             done = False
#             while not done:
                
#                 logits = policymodel(np.expand_dims(s,0))
#                 action = tf.squeeze(tf.random.categorical(logits = logits, num_samples = 1))
#                 [newstate, reward, done, _] = env.step(action.numpy())
                
#                 batch_actions.append(action)
#                 batch_states.append(s.copy())
#                 batch_rewards.append(reward)
#                 s = newstate.copy()
            
#         np_batch_action = np.array(batch_actions, dtype = 'int32')
#         np_batch_states = np.array(batch_states),
#         np_batch_rewards = np.array(batch_rewards)
        
#         print(len(np_batch_states))

#         loss = lossfunc(model= policymodel, 
#                           batch_rew= np_batch_rewards, 
#                           batch_action=np_batch_action, 
#                           batch_states=np_batch_states, 
#                           n_action=2)
#         return loss
    

#     loss = train_one_batch(2)
#     return  loss




    # logit = compute_logits(model, s)
    # print(logit)

    # action = sample_action(model, s)
    # print(action.numpy())






                



    # for ibatch in range(n_batches):
    #     rew = []
    #     for iep in range(epochs):
    #         state = env.reset()
    #         done = False
    #         while not done:
    #             action = sample_action(policy, state)
    #             [state, reward, done, _] = env.step(action)
    #             rew.append(reward)
    #     j = 2

    # policy_logit = get_policy(state, action_space)
    # action = tf.random.categorical(logit = policy_logit, num_samples = 1)






