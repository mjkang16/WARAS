# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# epsilon-greedy 시, action별 선택될 확률 계산
def epsilonmax(x, eps):
    n = len(x)
    p = np.ones(n)*eps/n
    p[np.argmax(x)] += 1 - eps
    
    return p

# softmax 시, action별 선택될 확률 계산
def softmax(x, scale = 1):
    x = np.array(x)/scale
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    p = e_x/e_x.sum()
    p = p/p.sum()
    return p

# softmax 시, state-value V 계산
def softV(z, scale = 1.):
    z = np.array(z)/scale
    max_z = np.max(z, axis=1)
    e_z = np.exp(z - max_z[:, np.newaxis])
    e_sum = np.sum(e_z, axis=1)
    e_sum = scale * (np.log(e_sum) + max_z)
    return e_sum

# sparsemax 시, action별 선택될 확률 계산
def sparsedist(z, scale=1.):
    z = np.array(z/scale)
    if len(z.shape) == 1:
        z = np.reshape(z,(1,-1))
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    p = np.maximum(0, z - tau_z)
    
    return p

# sparsemax 시, state-value V 계산
def sparsemax(z, scale=1.):
    z = np.array(z/scale)
    x = np.mean(z, axis=1)
    z = z - x[:, np.newaxis]
    
    # calculate sum over S(z)
    p = sparsedist(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    return scale*(0.5 * S_sum + 0.5 + x)

# Exploration 기법에 따른 action 결정 함수
def choice_action(Exp, eps, scale, action_Q):
    if Exp == 'softmax':
        action_max = softmax(action_Q, scale)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    elif Exp == 'sparsemax':
        action_dist = sparsedist(action_Q, scale)[0]
        n = len(action_dist)
        action_dist = np.ones(n) * eps/n + action_dist*(1.-eps)
        action_dist = action_dist/np.sum(action_dist)
        return np.random.choice(len(action_dist),size=1,p=action_dist)[0]
    elif Exp == 'epsilon':
        action_max = epsilonmax(action_Q, eps)
        return np.random.choice(len(action_max),size=1,p=action_max)[0]
    else:
        print("Exploration Type is wrong.")
        return 0
    
        