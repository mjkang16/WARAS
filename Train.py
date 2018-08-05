# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import Exploration
import time

GAMMA = 0.99

# Prioritized==True && DDPG==False 일 때 실행
def train_prioritized(Q_Network, train_batch, w_batch, Exp, s_scale, input_size, num_actions, size_action_batch):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    batch_size = len(done_batch)
    q_t_1_batch = []
    
    if Exp == 'epsilon':
        # 'target-Q네트워크'로 계산한 action별 q값 계산
        q_t_1_batch = Q_Network.get_target_q_batch(state_t_1_batch)
        q_t_1_batch = np.reshape(q_t_1_batch, [batch_size, -1])
        
        # Exploration 기법 별로 V(next_state)를 계산
        q_t_1_batch = np.max(q_t_1_batch, axis = 1)
        # 측정 Q(state, action)을 q_t_1_batch로 저장 
        q_t_1_batch = reward_batch + GAMMA*q_t_1_batch#*(1-done_batch)
        
    elif Exp == 'softmax':
        q_t_1_batch = Q_Network.get_target_q_batch(state_t_1_batch)
        q_t_1_batch = np.reshape(q_t_1_batch, [batch_size, -1])
        
        q_t_1_batch = Exploration.softV(q_t_1_batch, s_scale)
        q_t_1_batch = reward_batch + GAMMA*q_t_1_batch#*(1-done_batch)
    
    elif Exp == 'sparsemax':
        q_t_1_batch = Q_Network.get_target_q_batch(state_t_1_batch)
        q_t_1_batch = np.reshape(q_t_1_batch, [batch_size, -1])
        
        q_t_1_batch = Exploration.sparsemax(q_t_1_batch, s_scale)
        q_t_1_batch = reward_batch + GAMMA*q_t_1_batch#*(1-done_batch)
        
    q_t_1_batch = np.reshape(q_t_1_batch,[-1,1])
    w_batch = np.reshape(w_batch,[-1,1])
    
    # q_t_1_batch와 오차가 가장 작아지는 방향으로 Q네트워크 training, weight 적용
    errors, cost, _ = Q_Network.train_critic_prioritized(state_t_batch, action_batch, q_t_1_batch, w_batch)
    errors = np.sum(errors, axis=1)
    
    return errors, cost, state_t_batch

# Prioritized==True && DDPG==True 일 때 실행
def train_prioritized_DDPG(Q_Network, Action_Network, train_batch, w_batch, num_actions, grad_inv):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    action_t_1_batch = Action_Network.evaluate_target_actor(state_t_1_batch)
    w_batch = np.reshape(w_batch,[-1,1])
    
    q_t_1 = Q_Network.evaluate_target_critic(state_t_1_batch, action_t_1_batch)
    q_t_1 = np.reshape(q_t_1, [1,-1])
    
    y_i_batch = reward_batch + GAMMA*q_t_1#*(1-done_batch)
    y_i_batch = np.reshape(y_i_batch,[-1,1])

    # Update critic by minimizing the loss
    errors, cost, _ = Q_Network.train_critic_prioritized(state_t_batch, action_batch, y_i_batch, w_batch)
    errors = np.sum(errors, axis=1)
    
    # Update actor proportional to the gradients
    action_for_delQ = Action_Network.evaluate_actor(state_t_batch)

    del_Q_a = Q_Network.compute_delQ_a(state_t_batch, action_for_delQ)
    del_Q_a = grad_inv.invert(del_Q_a, action_for_delQ)
    
    Action_Network.train_actor(state_t_batch, del_Q_a)

    return errors, cost

# Prioritized==False && DDPG==False 일 때 실행    
def train(Q_Network, train_batch, Exp, s_scale, input_size, num_actions, size_action_batch):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    q_t_1_batch = []
    
    if Exp == 'epsilon':
        
        for i in range(0, len(train_batch)):
            q_t_1_batch.append(np.reshape(Q_Network.get_target_q_batch(np.reshape(state_t_1_batch[i],[1,-1])),[1,-1])[0])
            
        q_t_1_batch = reward_batch + GAMMA*np.max(q_t_1_batch, axis=1)#*(1-done_batch)
        
    elif Exp == 'softmax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.softV(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    elif Exp == 'sparsemax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.sparsemax(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    q_t_1_batch = np.reshape(q_t_1_batch,[-1,1])
    
    cost, _ = Q_Network.train_critic(state_t_batch, action_batch, q_t_1_batch)
    
    return cost, state_t_batch

# Prioritized==False && DDPG==True 일 때 실행
def train_DDPG(Q_Network, Action_Network, train_batch, num_actions, grad_inv):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    action_t_1_batch = Action_Network.evaluate_target_actor(state_t_1_batch)
    
    q_t_1 = Q_Network.evaluate_target_critic(state_t_1_batch, action_t_1_batch)
    q_t_1 = np.reshape(q_t_1, [1,-1])
    
    y_i_batch = reward_batch + GAMMA*q_t_1#*(1-done_batch)
    y_i_batch = np.reshape(y_i_batch,[-1,1])

    # Update critic by minimizing the loss
    cost, _ = Q_Network.train_critic(state_t_batch, action_batch, y_i_batch)
    
    # Update actor proportional to the gradients:
    action_for_delQ = Action_Network.evaluate_actor(state_t_batch)

    del_Q_a = Q_Network.compute_delQ_a(state_t_batch, action_for_delQ)
    del_Q_a = grad_inv.invert(del_Q_a, action_for_delQ)
    
    Action_Network.train_actor(state_t_batch, del_Q_a)

    return cost
    
