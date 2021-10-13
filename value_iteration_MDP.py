import numpy as np
from math import factorial
import time
import pickle

'''reward matrix'''


def poisson_pdf(lba, n):
    ''' calculate poisson pdf '''
    return np.exp(- lba) * (lba ** n) / factorial(n)

def poisson_cdf(lba,n):
    '''P(X<=n)'''
    return sum([poisson_pdf(lba,i) for i in range(n + 1)])


def cal_transition_prob(lambda_request, lambda_return, old_state, new_state):
    # calculate transition_prob of
    diff = new_state - old_state
    prob = 0
    
    if new_state == max_car_num:
        state_range = range(0,old_state)
        for num in state_range:
            prob += poisson_pdf(lambda_request,num) * ( 1 - poisson_cdf(lambda_return, num + diff - 1))
        prob += ( 1 - poisson_cdf(lambda_request,old_state - 1)) * ( 1 - poisson_cdf(lambda_return, new_state - 1))
    else:
        state_range = range(abs(min(0, new_state - old_state)), old_state)
        for num in state_range:
            prob += poisson_pdf(lambda_request, num) * poisson_pdf(lambda_return, num + diff)
        
        prob += ( 1 - poisson_cdf(lambda_request, old_state - 1)) * poisson_pdf(lambda_return, new_state)
    
    return prob


def transition_function(init_state, action, new_state):
    global max_car_num, max_move_num
    global lambda_request_A, lambda_request_B, lambda_return_A, lambda_return_B
    
    if not (action >= -min(max_move_num, init_state[1]) and action <= min(max_move_num, init_state[0])):
        # print('!',action)
        return 0
    
    old_state_A, old_state_B = min(max_car_num, init_state[0] - action), min(max_car_num, init_state[1] + action)
    new_state_A, new_state_B = new_state[0], new_state[1]
    
   
    transition_prob_A = cal_transition_prob(lambda_request_A, lambda_return_A, old_state_A, new_state_A)
    transition_prob_B = cal_transition_prob(lambda_request_B, lambda_return_B, old_state_B, new_state_B)
    
    return transition_prob_A * transition_prob_B


def cal_conditional_expectation(state, lambda_request, per_reward):
    valid_request = per_reward * sum([n * poisson_pdf(lambda_request, n) for n in range(0, state + 1)])
    exceed_request = per_reward * state * (1 - sum([poisson_pdf(lambda_request, n) for n in range(0, state + 1)]))
    
    return valid_request + exceed_request


def reward_function(old_state, action, action_fee):
    global max_car_number
    global per_reward
    
    old_state_A, old_state_B = old_state[0], old_state[1]
    
    if not (action >= -min(max_move_num, old_state_B) and action <= min(max_move_num, old_state_A)):
        print(old_state_A, old_state_B, action)
        # return None
        return 0
    
    new_state_A, new_state_B = min(max_car_num, old_state[0] - action), min(max_car_num, old_state[1] + action)
    
    cost = action_fee * abs(action)
    
    reward_A = cal_conditional_expectation(new_state_A, lambda_request_A, per_reward)
    reward_B = cal_conditional_expectation(new_state_B, lambda_request_B, per_reward)
    
    total_reward = reward_B + reward_A - cost
    
    return total_reward




def cal_value_function(cur_state, state_value, reward_pkl, temp_trans_prob_pkl):
    
    state_A, state_B = cur_state
    action_range = range(-5, 6)
    action_value_list = np.zeros(11)
    for iax, action in enumerate(action_range):
        cur_reward = reward_pkl[iax][state_A][state_B]
        value = cur_reward
        for isa, new_state_A in enumerate(range(max_car_num + 1)):
            for isb, new_state_B in enumerate(range(max_car_num + 1)):
                cur_tran_prob = temp_trans_prob_pkl[action + 5][new_state_A][new_state_B]
                value += discount_factor * cur_tran_prob * state_value[new_state_A][new_state_B]
        
        action_value_list[iax] = value
    
    return np.max(action_value_list),np.argmax(action_value_list) - max_move_num

def value_iteration(state_value, discount_factor, theta, max_car_num, reward_pkl, trans_prob_pkl):
    delta = 1
    while delta > theta:
        delta = 0
        start = time.time()
        for state_A in range(0, max_car_num + 1):
            for state_B in range(0, max_car_num + 1):
                v = state_value[state_A, state_B]
                temp_trans_prob_pkl = trans_prob_pkl[(state_A, state_B)]
                best_action_val = cal_value_function([state_A, state_B], state_value, reward_pkl, temp_trans_prob_pkl)[
                    0]
                delta = max(delta, abs(v - best_action_val))
                state_value[state_A, state_B] = best_action_val
        print('delta:',delta, 'running time:',time.time() - start)
    return state_value


if __name__ == '__main__':
    # V(3,3),V(0,2)
    max_car_num = 20
    max_move_num = 5
    theta = 0.1
    discount_factor = 0.9
    per_reward = 10
    action_fee = 2
    lambda_request_A, lambda_request_B, lambda_return_A, lambda_return_B = 3, 4, 3, 2

    # #  calculate reward matrix and transition matrix into pkl file
    # reward = np.zeros((11, 21, 21))
    # for iax, action in enumerate(range(-5, 6)):
    #     for isa, state_A in enumerate(range(21)):
    #         for isb, state_B in enumerate(range(21)):
    #             reward[action + 5, isa, isb] = reward_function([state_A, state_B], action, action_fee)
    #
    # fo = open('C:\\Users\\21130\\Desktop\\RL\\codes\\SUOSUO\\temp_reward', 'wb')
    # pickle.dump(reward, fo)
    # fo.close()
    #
    # trans_prob = {}
    # for isa, state_A in enumerate(range(21)):
    #     for isb, state_B in enumerate(range(21)):
    #         prob = np.zeros((11, 21, 21))
    #         for iax, action in enumerate(range(-5, 6)):
    #             for isna, new_state_A in enumerate(range(21)):
    #                 for isnb, new_state_B in enumerate(range(21)):
    #                     prob[iax, new_state_A, new_state_B] = transition_function([state_A, state_B], action,
    #                                                                               [new_state_A, new_state_B])
    #         trans_prob[(state_A, state_B)] = prob
    # #
    # po = open('C:\\Users\\21130\\Desktop\\RL\\codes\\SUOSUO\\temp_prob', 'wb')
    # pickle.dump(trans_prob, po)
    # po.close()

    fr = open('temp_reward', 'rb')
    reward_pkl = pickle.load(fr)
    pr = open('temp_prob', 'rb')
    trans_prob_pkl = pickle.load(pr)


    for i in range(-5, 6):
        print('r([3,3],', i, ")", '=', reward_function([3, 3], i, 2))
    
    for i in range(-5, 6):
        print('P([3,3]|[3,3],', i, ")", '=', transition_function([3, 3], i, [3, 3]))

    state_value = np.zeros((max_car_num + 1, max_car_num + 1))
    policy = np.zeros((max_car_num + 1, max_car_num + 1))
    state_value = value_iteration(state_value, discount_factor, theta, max_car_num, reward_pkl, trans_prob_pkl)
    
    
    policy = np.zeros((max_car_num + 1, max_car_num + 1))
    for state_A in range(max_car_num + 1):
        for state_B in range(max_car_num + 1):
            temp_trans_prob_pkl = trans_prob_pkl[(state_A, state_B)]
            best_action = cal_value_function([state_A, state_B], state_value, reward_pkl, temp_trans_prob_pkl)[1]
            policy[state_A, state_B] = best_action
    
    print('V[3,3]', state_value[3, 3])
    for i in range(max_car_num + 1):
        print('pi[{},3] is {}'.format(i,policy[i,3]))
    print('pi[3,3]', policy[3,3])

    import matplotlib.pyplot as  plt

    plt.contour(range(0, 21), range(0, 21), policy.T, colors='black')
    # plt.clabel(C, inline=True, fontsize=10)
    plt.show()
    #
    import seaborn as sns
    ax = sns.heatmap(policy, cmap='RdBu')
    ax.invert_yaxis()
    plt.show()



