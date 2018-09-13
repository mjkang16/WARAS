# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random
from collections import deque
import gym
from Q_Network import Q_Network
from Action_Network import Action_Network
from tensorflow_grad_inverter import grad_inverter
import Train
import Exploration
import replaymemory
import matplotlib.pyplot as plt
import pickle
import time

class RL:
    def __init__(self, sess, dis = 0.99, REPLAY_MEMORY = 100000, batch_size = 64, max_steps = 10000000, max_episodes = 500000,
                 
                 layer_size_Q1 = 300, layer_size_Q2 = 400, learning_rate_Q = 0.0001, learning_rate_A = 0.01,
                 training_step = 1, copy_step = 1, action_copy_step = 1, repu_num = 1,
                 beta_max_step = 1500000, action_res = None, random_action = True, weighted_train = False,
                 
                 ending_cond_epis = 100, ending_cond_reward = 195,
                 alpha = 0.6, beta_init = 0.4, eps = 0.01, eps_div = 256, s_scale = 1, size_action_batch = 500,
                 
                 Double = True, Prioritized = True, Exp = 'softmax', DDPG = False, action_train = False,
                 seed_n = 0, Game = 'CartPole-v0', file_name = 'steps', save_epi = 100, save_network = False):
        
        env = gym.make(Game)
        tf.set_random_seed(seed_n)
        rng = np.random.RandomState(seed_n)
        
        input_size = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        output_size = self.action_dim
        
        action_map = []
        action_scale = env.action_space.high[0]
        """
        for o in range(self.action_dim):
            print(env.action_space.low[o])
            print(env.action_space.high[o])
        """
        # DDPG 시, Action Network 생성
        if DDPG:
            self.Action_Network = Action_Network(sess, input_size, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_A, action_scale = action_scale)
        # Action이 랜덤하지 않을 시, get_action_information을 사용해 균일하게 action 범위를 자름
        if not random_action:
            size_action_batch, conti_action_flag, action_map = self.get_action_information(env, Game, action_res=action_res)
        
        replay_memory = replaymemory.ReplayMemory(rng=rng, memory_size=REPLAY_MEMORY, per_alpha=alpha, per_beta0=beta_init)
        
        ############### parameter 복사 ################
        self.sess = sess
        self.dis = dis
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.replay_memory_size = REPLAY_MEMORY
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.size_action_batch = size_action_batch
        
        self.alpha = alpha
        self.beta_init = beta_init
        self.beta_max_step = beta_max_step
        self.eps = eps
        self.eps_div = eps_div
        self.s_scale = s_scale
        self.action_scale = action_scale
        
        self.layer_size_Q1 = layer_size_Q1
        self.layer_size_Q2 = layer_size_Q2
        self.learning_rate_Q = learning_rate_Q
        self.learning_rate_A = learning_rate_A
        
        self.training_step = training_step
        self.copy_step = copy_step
        self.action_copy_step = action_copy_step
        self.action_train = action_train
        self.weighted_train = weighted_train
        self.repu_num = repu_num
        
        self.Double = Double
        self.Prioritized = Prioritized
        self.Exp = Exp
        self.DDPG = DDPG
        
        self.seed_n = seed_n
        self.Game = Game
        self.save_epi = save_epi
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.env = env
        self.file_name = file_name
        self.save_network = save_network
        self.random_action = random_action
        
        self.input_size = input_size
        self.output_size = output_size
        self.ending_cond_epis = ending_cond_epis
        self.ending_cond_reward = ending_cond_reward
        #################################################
        
        self.Q_Network = Q_Network(sess, input_size, output_size, size_action_batch, batch_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q, learning_rate_A, action_map, random_action, action_scale)
        
        # run_DQN 실행
        self.run_DQN(seed_n = seed_n, Exp = Exp, Double = Double, Prioritized = Prioritized)
    
    def get_action_information(self, env, env_name, action_res=None):
        action_map = []
        if isinstance(env.action_space, gym.spaces.Box):
            conti_action_flag = True
            if env_name == "Pendulum-v0" or env_name == "InvertedPendulum-v1" or env_name == "MountainCarContinuous-v0" or env_name == "InvertedDoublePendulum-v1":
                #action_map = np.linspace(env.action_space.low[0],env.action_space.high[0],num=action_res)
                action_map = np.zeros([np.prod(action_res), 1])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                for i in range(action_res[0]):
                    action_map[i, :] = [u[i]]
            
            elif env_name == "Reacher-v1":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Swimmer-v1" or env_name == "LunarLanderContinuous-v2" or env_name == "MultiGoal-v0":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Hopper-v1":
                action_map = np.zeros([np.prod(action_res), 3])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                w = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        for k in range(action_res[2]):
                            s = action_res[2] * action_res[1] * i + action_res[2] * j + k
                            action_map[s, :] = [u[i], v[j], w[k]]
            
            elif env_name == "Walker2d-v1" or env_name == "HalfCheetah-v1":
                action_map = np.zeros([np.prod(action_res), 6])
                x = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                y = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                z = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                u = np.linspace(env.action_space.low[3], env.action_space.high[3], num=action_res[3])
                v = np.linspace(env.action_space.low[4], env.action_space.high[4], num=action_res[4])
                w = np.linspace(env.action_space.low[5], env.action_space.high[5], num=action_res[5])
                for i0 in range(action_res[0]):
                    for i1 in range(action_res[1]):
                        for i2 in range(action_res[2]):
                            for i3 in range(action_res[3]):
                                for i4 in range(action_res[4]):
                                    for i5 in range(action_res[5]):
                                        s = np.prod(action_res[1:]) * i0
                                        s += np.prod(action_res[2:]) * i1
                                        s += np.prod(action_res[3:]) * i2
                                        s += np.prod(action_res[4:]) * i3
                                        s += np.prod(action_res[5:]) * i4
                                        s += i5
                                        action_map[s, :] = [x[i0], y[i1], z[i2], u[i3], v[i4], w[i5]]
            elif env_name == "Ant-v1":
                action_map = np.zeros([np.prod(action_res), 8])
                x = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                y = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                z = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                u = np.linspace(env.action_space.low[3], env.action_space.high[3], num=action_res[3])
                v = np.linspace(env.action_space.low[4], env.action_space.high[4], num=action_res[4])
                w = np.linspace(env.action_space.low[5], env.action_space.high[5], num=action_res[5])
                p = np.linspace(env.action_space.low[6], env.action_space.high[6], num=action_res[6])
                q = np.linspace(env.action_space.low[7], env.action_space.high[7], num=action_res[7])
                for i0 in range(action_res[0]):
                    for i1 in range(action_res[1]):
                        for i2 in range(action_res[2]):
                            for i3 in range(action_res[3]):
                                for i4 in range(action_res[4]):
                                    for i5 in range(action_res[5]):
                                        for i6 in range(action_res[6]):
                                            for i7 in range(action_res[7]):
                                                s = np.prod(action_res[1:]) * i0
                                                s += np.prod(action_res[2:]) * i1
                                                s += np.prod(action_res[3:]) * i2
                                                s += np.prod(action_res[4:]) * i3
                                                s += np.prod(action_res[5:]) * i4
                                                s += np.prod(action_res[6:]) * i5
                                                s += np.prod(action_res[7:]) * i6
                                                s += i7
                                        action_map[s, :] = [x[i0], y[i1], z[i2], u[i3], v[i4], w[i5], p[i6], q[i7]]
            else:
                print(env.action_space.high.shape[0])
            n_action = np.prod(action_res)
        
        elif isinstance(env.action_space, gym.spaces.Discrete):
            conti_action_flag = False
            n_action = env.action_space.n
        
        else:
            raise NotImplementedError("{} action spaces are not supported yet.".format(type(env.action_space)))
        return n_action, conti_action_flag, action_map
    
    # Calculate the action distance
    # 모든 행동 간의 거리를 측정한 뒤, 모든 행동 간의 평균 거리와 특정거리보다 가까운 쌍의 개수를 세는 함수
    def get_action_variance(self, A_batch):
        A_batch = np.array(A_batch)
        num_actions = np.shape(A_batch)[0]
        dim_actions = np.shape(A_batch)[1]
        
        distance = 0.
        num = 0.
        num2 = 0.
        for i in range(num_actions):
            A = np.square(A_batch - A_batch[i])
            A = np.sum(A, axis = 1)
            
            num = num + (A < 0.3).sum()
            num2 = num2 + (A < 1).sum()
            
            A = np.sqrt(A)
            distance = distance + np.sum(A)
        
        num = (num-num_actions)*100/(num_actions*(num_actions-1))
        num = round(num, 3)
        
        num2 = (num2-num_actions)*100/(num_actions*(num_actions-1))
        num2 = round(num2, 3)
        
        distance = distance / (num_actions*(num_actions-1))
        distance = round(distance, 4)
        
        # distance : 모든 행동간 평균 거리
        # num : 전체 행동 쌍 중에서 거리가 루트(0.3)미만인 행동 쌍의 비율
        # num2 : 전체 행동 쌍 중에서 거리가 루트(1)미만인 행동 쌍의 개수
        return distance, num, num2
    
    # resampling the action batch
    def realign_action_batch(self, A_batch, t_A_batch):
        A_batch = np.array(A_batch)
        t_A_batch = np.array(t_A_batch)
        num_actions = np.shape(A_batch)[0]
        dim_actions = np.shape(A_batch)[1]
        
        for i in range(num_actions):
            A = np.square(A_batch - A_batch[i])
            A = np.sum(A, axis = 1)
            
            # A_batch[i]와 거리가 루트(0.3)미만인 행동(A_batch[i]를 제외한)이 존재할 시,
            if (A < 0.3).sum() > 1:
                A_batch[i] = np.random.random(dim_actions)*2*self.action_scale - self.action_scale # resampling
                t_A_batch[i] = A_batch[i]                                                          # t_A_batch[i]도 교체
            
        return A_batch, t_A_batch
    
    def format_experience(self, experience):
        states_b, actions_b, rewards_b, states_n_b, done_b = zip(*experience)
        
        minibatch = []
        for num in range(len(states_b)):
            minibatch.append((states_b[num],states_n_b[num],actions_b[num],rewards_b[num],done_b[num]))
        return minibatch
    
    def run_DQN(self, seed_n, Exp, Double, Prioritized):
        ############## parameter 복사 ##############
        sess = self.sess
        dis = self.dis
        REPLAY_MEMORY = self.REPLAY_MEMORY
        replay_memory = self.replay_memory
        batch_size = self.batch_size
        size_action_batch = self.size_action_batch

        Game = self.Game
        save_epi = self.save_epi
        save_network = self.save_network
        max_episodes = self.max_episodes
        max_steps = self.max_steps
        env = self.env
        random_action = self.random_action
        
        input_size = self.input_size
        output_size = self.output_size
        
        alpha = self.alpha
        beta_init = self.beta_init
        beta_max_step = self.beta_max_step
        eps = self.eps
        eps_div = self.eps_div
        s_scale = self.s_scale
        
        training_step = self.training_step
        copy_step = self.copy_step
        action_copy_step = self.action_copy_step
        action_train = self.action_train
        weighted_train = self.weighted_train
        repu_num = self.repu_num
        
        DDPG = self.DDPG
        
        ending_cond_epis = self.ending_cond_epis
        ending_cond_reward = self.ending_cond_reward
        
        env.seed(seed_n)
        np.random.seed(seed_n)
        tf.set_random_seed(seed_n)
        random.seed(seed_n)
        #############################################
        
        Q_Network = self.Q_Network
        A_batch = Q_Network.get_action_batch()
        if DDPG:
            Action_Network = self.Action_Network
        
        # DDPG Action Network 학습 시 사용되는 grad_inv 설정
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()        
        action_bounds = [action_max,action_min] 
        grad_inv = grad_inverter(sess, action_bounds)
        
        case_n = seed_n + 1
        end_episode = 0
        step_count_total = 0
        global_step = 0
        loss = 0
        e = 1.

        replay_buffer = deque()
        Q_list = []
        TD_buffer = deque()
        steps_list = []
        step_avg_list = []
        global_step_list = []
        
        average_distance = []
        rate_of_adjacent = []

        print("")
        print("CASE {}".format(case_n))
        print("  STATE DIM : {}, ACTION DIM : {}".format(input_size, self.action_dim))
        print("  Exp : {}".format(Exp))
        if DDPG:
            print("  Strategy : Double : {},  Prioritized : {},  DDPG : {}".format(Double, Prioritized, DDPG))
        elif random_action:
            if action_train:
                print("  Strategy : Double : {},  Prioritized : {},  ACTION : RANDOM,  ACTION TRAIN 'ON'".format(Double, Prioritized))
            else:
                print("  Strategy : Double : {},  Prioritized : {},  ACTION : RANDOM".format(Double, Prioritized))
        else:
            if action_train:
                print("  Strategy : Double : {},  Prioritized : {},  ACTION : DISCRETIZATION,  ACTION TRAIN 'ON'".format(Double, Prioritized))
            else:
                print("  Strategy : Double : {},  Prioritized : {},  ACTION : DISCRETIZATION".format(Double, Prioritized))
        print("")
        
        for episode in range(1, max_episodes+1):
            
            done = False
            step_count = 0
            current_step = 0
            cost = 0
            state = env.reset()
            
            while not done:
                # 입실론 값 조정, 0.001미만이 될 시 더 이상 작아지지 않는다.
                if e > 0.001:
                    #e = 1. / ((float(episode - 1) / eps_div) + 1)
                    e = 1. / ((float(global_step) / eps_div) + 1)
                
                t4 = time.time()
                if DDPG: # DDPG true 시, 액션네트워크로부터 행동을 결정받음
                    action = Action_Network.evaluate_actor(np.reshape(state, [1, input_size]))[0]
                else: # DDPG false 시, state에 따른 각 행동 별 q 값을 get_q_batch로 받은 후 Exploration 방식에 따라 행동 결정
                    action0 = Exploration.choice_action(Exp, e, s_scale,\
                                                 np.reshape(Q_Network.get_q_batch(np.reshape(state,[1,-1])),[1,-1])[0])
                    action = A_batch[action0]
                
                next_state, reward, done, _ = env.step(action)
                step_count += reward
                global_step += 1
                current_step += 1
                
                # Prioritized 시 tree(replay_memory)에 저장, 아닐 시 랜덤으로 추출할 replay_beffer에 저장
                if Prioritized:
                    replay_memory.save_experience(state, action, reward, next_state, done)
                else:
                    replay_buffer.append((state, next_state, action, reward, done))
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()
                
                state = next_state
                
                if global_step <= beta_max_step:
                    replay_memory.anneal_per_importance_sampling(global_step, beta_max_step)
                
                # training step마다 traing 실행
                if global_step > batch_size and global_step % training_step == 0:
                    for re in range(repu_num): # repu_num만큼 반복 training. 거의 1로 사용.
                        if Prioritized:
                            # replay_memory로부터 batch를 추출
                            idx, priorities, w_batch, experience = replay_memory.retrieve_experience(batch_size)
                            minibatch = self.format_experience(experience)
                            if DDPG:
                                # DDPG true시 Q네트워크와 Action네트워크 모두 training
                                errors, cost = Train.train_prioritized_DDPG(Q_Network, Action_Network, minibatch, w_batch, output_size, grad_inv)
                                replay_memory.update_experience_weight(idx, errors)
                            
                            else:
                                # DDPG false시 Q네트워크 training
                                errors, cost, state_t_batch = Train.train_prioritized(Q_Network, minibatch, w_batch, Exp, s_scale, input_size, output_size, size_action_batch)
                                replay_memory.update_experience_weight(idx, errors)
                                
                                # action_copy_step 마다 action set을 training, action_train이 false 시 RAS 알고리즘
                                if action_train and global_step % action_copy_step == 0:
                                    action_weight = []
                                    
                                    
                                    if weighted_train: # WARAS 알고리즘
                                        # weight 계산
                                        for k in range(batch_size):
                                            state_t = np.reshape(state_t_batch[k], [1,-1])
                                            
                                            q_batch = Q_Network.get_q_batch(state_t)
                                            q_batch = np.reshape(q_batch, [1,-1])[0]
                                            q_batch = q_batch*10.
                                            max_q = np.max(q_batch)
                                            q_batch = np.exp(q_batch - max_q)
                                            action_weight.append(q_batch)
                                            
                                    else: # ARAS 알고리즘
                                        # 모든 weight를 1로 설정
                                        action_weight = np.ones([batch_size,size_action_batch])
                                    # weight 값을 이용한 Q네트워크 training
                                    Q_Network.train_weighted_actor(state_t_batch, action_weight)
                                    
                                    # target-action set을 update
                                    Q_Network.update_action_target_critic()
                                    A_batch = Q_Network.get_action_batch()
                                    t_A_batch = Q_Network.get_target_action_batch()
                                    """
                                    # 거리가 가까운 action 쌍을 찾아 resampling
                                    A_batch, t_A_batch = self.realign_action_batch(A_batch, t_A_batch)
                                    Q_Network.realign_action_batch(A_batch, t_A_batch)
                                    A_batch = Q_Network.get_action_batch()
                                    t_A_batch = Q_Network.get_target_action_batch()
                                    """
                        else: # Prioritized가 아닐 시 랜덤하게 minibatch를 생성해 training
                            minibatch = random.sample(replay_buffer, batch_size)
                            if DDPG:
                                cost = Train.train_DDPG(Q_Network, Action_Network, minibatch, output_size, grad_inv)
                            
                            else:
                                cost, state_t_batch = Train.train(Q_Network, minibatch, Exp, s_scale, input_size, output_size, size_action_batch)
                
                # copy_step 마다 Q네트워크 업데이트
                if global_step % copy_step == 0:
                    if DDPG:
                        # Update target Critic and actor network
                        Q_Network.update_target_critic()
                        Q_Network.update_action_target_critic()
                        Action_Network.update_target_actor()
                    
                    else:
                        Q_Network.update_target_critic()
                        Q_Network.update_action_target_critic()
                            
            steps_list.append(step_count)
            global_step_list.append(global_step)
            
            # Print the average of result 
            if episode < ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / episode)

            if episode == ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / ending_cond_epis)

            if episode > ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                step_avg_list.append(step_count_total / ending_cond_epis)
            
            print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
            if DDPG:
                print ("                   ( Result : {},  Loss : {},  Steps : {},  Global Steps : {} )"
                                   #.format(round(step_count, 3), round(cost, 5), current_step, global_step))
                                   .format(round(step_count, 3), 0, current_step, global_step))
            elif Exp == 'epsilon' or Exp == 'sparsemax':
                print ("                   ( Result : {},  Loss : {},  Epsilon : {},  Steps : {},  Global Steps : {} )"
                                   #.format(round(step_count, 3), round(cost, 5), round(e, 4), current_step, global_step))
                                   .format(round(step_count, 3), 0, round(e, 5), current_step, global_step))
            else:
                print ("                   ( Result : {},  Loss : {},  Steps : {},  Global Steps : {} )"
                                   #.format(round(step_count, 3), round(cost, 5), current_step, global_step))
                                   .format(round(step_count, 3), 0, current_step, global_step))
            
            distance, per_of_sim, per_of_sim2 = self.get_action_variance(A_batch)
            print ("                   ( Action Batch  ::::  Distance : {},  Percent : {}%({}%) )"
                                   .format(distance, per_of_sim, per_of_sim2))
            average_distance.append(distance)
            rate_of_adjacent.append(per_of_sim)
            
            # Save the networks 
            if episode % save_epi == 0:
                file_case = str(case_n)
                if save_network:
                    Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
            
                with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
                
                with open('/home/minjae/Desktop/JOLP/'+'Average_of_Distance_('+self.file_name+'_seed'+file_case+')', 'wb') as fout:
                    pickle.dump(average_distance, fout)
                with open('/home/minjae/Desktop/JOLP/'+'Rate_of_Adjacent_('+self.file_name+'_global_'+'_seed'+file_case+')', 'wb') as fout2:
                    pickle.dump(rate_of_adjacent, fout2)
                
                p_values = list(range(1, episode+1))
                q_values = average_distance[:]
                r_values = rate_of_adjacent[:]
                plt.plot(p_values, q_values, c='r')
                plt.title('Average of Distance between Actions')
                plt.grid(True)
                plt.show()
                plt.plot(p_values, r_values, c='b')
                plt.title('Rate of Adjacent Actions')
                plt.grid(True)
                plt.show()
                
            end_episode += 1
            
            # 결과가 목표치를 달성하면 학습 중단
            if step_avg_list[episode - 1] > ending_cond_reward:
                break
            # max_steps 만큼 학습되었으면 학습 중단    
            if global_step > max_steps:
                break

        print("--------------------------------------------------")
        print("--------------------------------------------------")
        
        # 목표치를 달성하여 학습 중단 시, 남은 episode 만큼 실행
        for episode in range(end_episode + 1, max_episodes+1):
            
            if global_step > max_steps:
                break
            
            s = env.reset()
            reward_sum = 0
            done = False
            while not done :
                # 최대 Q 값을 나타내는 행동 선택
                action = np.argmax(Q_Network.evaluate_critic(np.reshape(state, [1, input_size])))

                if conti_action_flag:
                    action = [action_map[action]]
                else:
                    action = action
                
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                global_step += 1

                if done :
                    steps_list.append(reward_sum)
                    global_step_list.append(global_step)
                    step_count_total += steps_list[episode - 1]
                    step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                    step_avg_list.append(step_count_total / ending_cond_epis)
                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {} )".format(reward_sum))
        
            if episode % save_epi == 0:
                file_case = str(case_n)
                if save_network:
                    Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
                with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
        
        # parameter 저장
        file_case = str(case_n)
        with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_seed'+file_case, 'wb') as fout:
            pickle.dump(step_avg_list, fout)
        with open('/home/minjae/Desktop/JOLP/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
            pickle.dump(global_step_list, fout2)
        
        # 그래프 출력
        x_values = list(range(1, len(step_avg_list)+1))
        y_values = step_avg_list[:]
        plt.plot(x_values, y_values, c='green')
        plt.title(self.file_name)
        plt.grid(True)
        plt.show()
        
        with open('/home/minjae/Desktop/JOLP/'+'Average_of_Distance_('+self.file_name+'_seed'+file_case+')', 'wb') as fout:
            pickle.dump(average_distance, fout)
        with open('/home/minjae/Desktop/JOLP/'+'Rate_of_Adjacent_('+self.file_name+'_global_'+'_seed'+file_case+')', 'wb') as fout2:
            pickle.dump(rate_of_adjacent, fout2)
        
        p_values = list(range(1, episode+1))
        q_values = average_distance[:]
        r_values = rate_of_adjacent[:]
        plt.plot(p_values, q_values, c='r')
        plt.title('Average of Distance between Actions')
        plt.grid(True)
        plt.show()
        plt.plot(p_values, r_values, c='b')
        plt.title('Rate of Adjacent Actions')
        plt.grid(True)
        plt.show()
        
        
        
