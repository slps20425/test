import tensorflow as tf
import numpy as np
from collections import deque
import random
import pandas as pd
import ffn

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#pip install pandas-datareader --upgrade

from pathlib import Path
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register





'''
data_dir = '/poloniex_data/'
directory = os.getcwd() + data_dir # path to the files
files_tags = os.listdir(directory) #these are the differents pdf files

#this is here because hidden files are also shown in the list. 
for file in files_tags:
    if file[0] == '.':
        files_tags.remove(file)
stock_name = [file.split('.')[0] for file in files_tags]
stocks = [file for file in files_tags]
print(len(stock_name) == len(stocks))
print('There are {} different currencies.'.format(len(stock_name)))


for s in stocks:
    df = pd.read_csv('.'+data_dir+s)
    print(s, len(df))

kept_stocks = ['ETCBTC.csv', 'ETHBTC.csv', 'DOGEBTC.csv', 'ETHUSDT.csv', 'BTCUSDT.csv', 
              'XRPBTC.csv', 'DASHBTC.csv', 'XMRBTC.csv', 'LTCBTC.csv', 'ETCETH.csv']
len_stocks = list()

for s in kept_stocks:
    df = pd.read_csv('.'+data_dir+s)
    len_stocks.append(len(df))

    
min_len = np.min(len_stocks)
min_len

list_open = list()
list_close = list()
list_high = list()
list_low = list()

for s in tqdm(kept_stocks):
    data = pd.read_csv(os.getcwd() + data_dir + s).fillna('bfill').copy()
    data = data[['open', 'close', 'high', 'low']]
    data = data.tail(min_len)
    list_open.append(data.open.values)
    list_close.append(data.close.values)
    list_high.append(data.high.values)
    list_low.append(data.low.values)

array_open = np.transpose(np.array(list_open))[:-1]
array_open_of_the_day = np.transpose(np.array(list_open))[1:]
array_close = np.transpose(np.array(list_close))[:-1]
array_high = np.transpose(np.array(list_high))[:-1]
array_low = np.transpose(np.array(list_low))[:-1]



'''












class TradeEnv():

    """
    This class is the trading environment (render) of our project. 
    The trading agent calls the class by giving an action at the time t. 
    Then the render gives back the new portfolio at the next step (time t+1). 
    #parameters:
    - windonw_length: this is the number of time slots looked in the past to build the input tensor
    - portfolio_value: this is the initial value of the portfolio 
    - trading_cost: this is the cost (in % of the traded stocks) the agent will pay to execute the action 
    - interest_rate: this is the rate of interest (in % of the money the agent has) the agent will:
        -get at each step if he has a positive amount of money 
        -pay if he has a negative amount of money
    -train_size: % of data taken for the training of the agent - please note the training data are taken with respect 
    of the time span (train -> | time T | -> test)
    """

    def __init__(self, path = './np_data/input.npy', window_length=30,
                 portfolio_value= 10000, trading_cost= 0.25/100,interest_rate= 0.02/250, train_size = 0.7):
        
        #path to numpy data
        self.path = path
        #load the whole data
        self.data = np.load(self.path)


        #parameters
        self.portfolio_value = portfolio_value
        self.window_length=window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        #number of stocks and features
        self.nb_stocks = self.data.shape[1]
        self.nb_features = self.data.shape[0]
        self.end_train = int((self.data.shape[2]-self.window_length)*train_size)
        
        #init state and index
        self.index = None
        self.state = None
        self.done = False

        #init seed
        self.seed()

    def return_pf(self):
        """
        return the value of the portfolio
        """
        return self.portfolio_value
        
    def readTensor(self,X,t):
        ## this is not the tensor of equation 18 
        ## need to batch normalize if you want this one 
        return X[ : , :, t-self.window_length:t ]
    
    def readUpdate(self, t):
        #return the return of each stock for the day t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, w_init, p_init, t=0 ):
        
        """ 
        This function restarts the environment with given initial weights and given value of portfolio
        """
        self.state= (self.readTensor(self.data, self.window_length) , w_init , p_init )
        self.index = t  +self.window_length
        self.done = False
        
        return self.state, self.done

    def step(self, action):
        """
        This function is the main part of the render. 
        At each step t, the trading agent gives as input the action he wants to do. So, he gives the new value of the weights of the portfolio. 
        The function computes the new value of the portfolio at the step (t+1), it returns also the reward associated with the action the agent took. 
        The reward is defined as the evolution of the the value of the portfolio in %. 
        """

        index = self.index
        #get Xt from data:
        data = self.readTensor(self.data, index)
        done = self.done
        
        #beginning of the day 
        state = self.state
        w_previous = state[1]
        pf_previous = state[2]
        

        #the update vector is the vector of the opening price of the day divided by the opening price of the previous day
        update_vector = self.readUpdate(index)

        #allocation choice 
        w_alloc = action
        pf_alloc = pf_previous
        
        #Compute transaction cost
        cost = pf_alloc * np.linalg.norm((w_alloc-w_previous),ord = 1)* self.trading_cost
        
        #convert weight vector into value vector 
        v_alloc = pf_alloc*w_alloc
        
        #pay transaction costs
        pf_trans = pf_alloc - cost
        v_trans = v_alloc - np.array([cost]+ [0]*self.nb_stocks)
        
        #####market prices evolution 
        #we go to the end of the day 
        
        #compute new value vector 
        v_evol = v_trans*update_vector

        
        #compute new portfolio value
        pf_evol = np.sum(v_evol)
        
        #compute weight vector 
        w_evol = v_evol/pf_evol
        w_evol =np.where(w_evol<0,0,w_evol)
        
        #compute instanteanous reward
        reward = (pf_evol-pf_previous)/pf_previous
        
        #update index
        index = index+1
        
        #compute state
        
        state = (data, w_evol, pf_evol) #self.readTensor(self.data, index) having problem 
        
        if index >= self.end_train:
            done = True
        
        self.state = state
        self.index = index
        self.done = done
        
        return state, reward, done

path_data ='/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/RLEPR/data/jiang_data.npy' #'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/1.5_preprocess_data/new_2021/jiang_input_45Tic_2012_2021.npy' #Path(__file__).with_name('jiang_input_45Tic.npy')
data = np.load(path_data)
trading_period = data.shape[2]
nb_feature_map = data.shape[0]
nb_stocks = data.shape[1]


# fix parameters of the network
m = nb_stocks

###############################dictionaries of the problem###########################
dict_hp_net = {'n_filter_1': 2, 'n_filter_2': 20, 'kernel1_size':(1, 3)}
dict_hp_pb = {'batch_size': 2**10, 'ratio_train': 0.8,'ratio_val': 0.2, 'length_tensor': 30,
              'ratio_greedy':0.5, 'ratio_regul': 0.1}
dict_hp_opt = {'regularization': 1e-8, 'learning': 9e-2}
dict_fin = {'trading_cost': 0.25/100, 'interest_rate': 0.02/250, 'cash_bias_init': 0.7}
dict_train = {'pf_init_train': 500000, 'w_init_train': 'd', 'n_episodes':10, 'n_batches':2}
dict_test = {'pf_init_test': 500000, 'w_init_test': 'd'}

list_stock =  ['1101','1102','1216','1301','1303','1326','1402','1590','2002','2207',\
 '2303','2308','2317','2327','2330','2357','2379','2382','2395','2408',\
 '2409','2412','2454','2603','2609','2615','2801','2880','2881','2882',\
 '2883','2884','2885','2886','2891','2892','2912','3008l','3034','3045',\
 '3481','4904','4938','6505','9910']
###############################HP of the network ###########################
n_filter_1 = dict_hp_net['n_filter_1']
n_filter_2 = dict_hp_net['n_filter_2']
kernel1_size = dict_hp_net['kernel1_size']

###############################HP of the problem###########################

# Size of mini-batch during training
batch_size = dict_hp_pb['batch_size']
# Total number of steps for pre-training in the training set
total_steps_train = 1827 # = int(dict_hp_pb['ratio_train']*trading_period)

# Total number of steps for pre-training in the validation set
total_steps_val = int(dict_hp_pb['ratio_val']*trading_period)

# Total number of steps for the test
total_steps_test = trading_period-total_steps_train-total_steps_val

# Number of the columns (number of the trading periods) in each input price matrix
n = dict_hp_pb['length_tensor']

ratio_greedy = dict_hp_pb['ratio_greedy']

ratio_regul = dict_hp_pb['ratio_regul']

##############################HP of the optimization###########################


# The L2 regularization coefficient applied to network training
regularization = dict_hp_opt['regularization']
# Parameter alpha (i.e. the step size) of the Adam optimization
learning = dict_hp_opt['learning']

optimizer =tf.train.AdamOptimizer(learning)
# optimizer = tf.optimizers.Adam(learning) # 


##############################Finance parameters###########################

trading_cost= dict_fin['trading_cost']
interest_rate= dict_fin['interest_rate']
cash_bias_init = dict_fin['cash_bias_init']

############################## PVM Parameters ###########################
sample_bias = 5e-5  # Beta in the geometric distribution for online training sample batches


############################## Training Parameters ###########################

w_init_train = np.array(np.array([1.0]+[.0]*m))#dict_train['w_init_train']

pf_init_train = dict_train['pf_init_train']

n_episodes = dict_train['n_episodes']
n_batches = dict_train['n_batches']

############################## Test Parameters ###########################

w_init_test = np.array(np.array([1.0]+[0]*m))#dict_test['w_init_test']

pf_init_test = dict_test['pf_init_test']


############################## other environment Parameters ###########################

w_eq = np.array(np.array([1/(m+1)]*(m+1)))

w_s = np.array(np.array([1]+[0.0]*m))


# random action function

def get_random_action(m):
    random_vec = np.random.rand(m+1)
    return random_vec/np.sum(random_vec)

#environment for trading of the agent 
# this is the agent trading environment (policy network agent)
env = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])


#environment for equiweighted
#this environment is set up for an agent who only plays an equiweithed portfolio (baseline)
env_eq = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])

#environment secured (only money)
#this environment is set up for an agentwho plays secure, keeps its money
env_s = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])

#full on one stock environment 
#these environments are set up for agents who play only on one stock

action_fu = list()
env_fu = list()


for i in range(m):
    action = np.array([0]*(i+1) + [1] + [0]*(m-(i+1)))
    action_fu.append(action)
    
    env_fu_i = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])
    
    env_fu.append(env_fu_i)

# define neural net \pi_\phi(s) as a class
class Policy(object):
    '''
    This class is used to instanciate the policy network agent

    '''

    def __init__(self, m, n, sess, optimizer,
                 trading_cost=trading_cost,
                 interest_rate=interest_rate,
                 n_filter_1=n_filter_1,
                 n_filter_2=n_filter_2):

        # parameters
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.n = n
        self.m = m

        with tf.variable_scope("Inputs"):

            # Placeholder

            # tensor of the prices
            self.X_t = tf.placeholder(
                tf.float32, [None, nb_feature_map, self.m, self.n])  # The Price tensor
            # weights at the previous time step
            self.W_previous = tf.placeholder(tf.float32, [None, self.m+1])
            # portfolio value at the previous time step
            self.pf_value_previous = tf.placeholder(tf.float32, [None, 1])
            # vector of Open(t+1)/Open(t)
            self.dailyReturn_t = tf.placeholder(tf.float32, [None, self.m])
            
            #self.pf_value_previous_eq = tf.placeholder(tf.float32, [None, 1])
            
            

        with tf.variable_scope("Policy_Model"):

            # variable of the cash bias
            bias = tf.get_variable('cash_bias', shape=[
                                   1, 1, 1, 1], initializer=tf.constant_initializer(cash_bias_init))
            # shape of the tensor == batchsize
            shape_X_t = tf.shape(self.X_t)[0]
            # trick to get a "tensor size" for the cash bias
            self.cash_bias = tf.tile(bias, tf.stack([shape_X_t, 1, 1, 1]))
            # print(self.cash_bias.shape)

            with tf.variable_scope("Conv1"):
                # first layer on the X_t tensor
                # return a tensor of depth 2
                self.conv1 = tf.layers.conv2d(
                    inputs=tf.transpose(self.X_t, perm=[0, 3, 2, 1]),
                    activation=tf.nn.relu,
                    filters=self.n_filter_1,
                    strides=(1, 1),
                    kernel_size=kernel1_size,
                    padding='same')

            with tf.variable_scope("Conv2"):
                
                #feature maps
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1,
                    activation=tf.nn.relu,
                    filters=self.n_filter_2,
                    strides=(self.n, 1),
                    kernel_size=(1, self.n),
                    padding='same')

            with tf.variable_scope("Tensor3"):
                #w from last periods
                # trick to have good dimensions
                w_wo_c = self.W_previous[:, 1:]
                w_wo_c = tf.expand_dims(w_wo_c, 1)
                w_wo_c = tf.expand_dims(w_wo_c, -1)
                self.tensor3 = tf.concat([self.conv2, w_wo_c], axis=3)

            with tf.variable_scope("Conv3"):
                #last feature map WITHOUT cash bias
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2,
                    activation=tf.nn.relu,
                    filters=1,
                    strides=(self.n_filter_2 + 1, 1),
                    kernel_size=(1, 1),
                    padding='same')

            with tf.variable_scope("Tensor4"):
                #last feature map WITH cash bias
                self.tensor4 = tf.concat([self.cash_bias, self.conv3], axis=2)
                # we squeeze to reduce and get the good dimension
                self.squeezed_tensor4 = tf.squeeze(self.tensor4, [1, 3])

            with tf.variable_scope("Policy_Output"):
                # softmax layer to obtain weights
                self.action = tf.nn.softmax(self.squeezed_tensor4)
                #tf.nn.softmax(self.squeezed_tensor4)
                
            with tf.variable_scope("Reward"):
                # computation of the reward
                # please look at the chronological map to understand
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
                Vprime_t = self.action * self.pf_value_previous
                Vprevious = self.W_previous*self.pf_value_previous

                # this is just a trick to get the good shape for cost
                constant = tf.constant(1.0, shape=[1])

                cost = self.trading_cost * \
                    tf.norm(Vprime_t-Vprevious, ord=1, axis=1)*constant

                cost = tf.expand_dims(cost, 1)

                zero = tf.constant(
                    np.array([0.0]*m).reshape(1, m), shape=[1, m], dtype=tf.float32)

                vec_zero = tf.tile(zero, tf.stack([shape_X_t, 1]))
                vec_cost = tf.concat([cost, vec_zero], axis=1)

                Vsecond_t = Vprime_t - vec_cost

                V_t = tf.multiply(Vsecond_t, y_t)
                self.portfolioValue = tf.norm(V_t, ord=1)
                self.instantaneous_reward = (
                    self.portfolioValue-self.pf_value_previous)/self.pf_value_previous
                
                
            with tf.variable_scope("Reward_Equiweighted"):
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
  

                V_eq = w_eq*self.pf_value_previous
                V_eq_second = tf.multiply(V_eq, y_t)
        
                self.portfolioValue_eq = tf.norm(V_eq_second, ord=1)
            
                self.instantaneous_reward_eq = (
                    self.portfolioValue_eq-self.pf_value_previous)/self.pf_value_previous
                
            with tf.variable_scope("Max_weight"):
                self.max_weight = tf.reduce_max(self.action)
                print(self.max_weight.shape)

                
            with tf.variable_scope("Reward_adjusted"):
                
                self.adjested_reward = self.instantaneous_reward - self.instantaneous_reward_eq - ratio_regul*self.max_weight
                
        # objective function 
        # maximize reward over the batch 
        # min(-r) = max(r)
        self.train_op = optimizer.minimize(-self.adjested_reward)
        
        # some bookkeeping
        self.optimizer = optimizer
        self.sess = sess
        tf.summary.FileWriter("Jiang_replicate/logs/", self.sess.graph)

    def compute_W(self, X_t_, W_previous_):
        """
        This function returns the action the agent takes 
        given the input tensor and the W_previous
        
        It is a vector of weight

        """

        return self.sess.run(tf.squeeze(self.action), feed_dict={self.X_t: X_t_, self.W_previous: W_previous_})

    def train(self, X_t_, W_previous_, pf_value_previous_, dailyReturn_t_):
        """
        This function trains the neural network
        maximizing the reward 
        the input is a batch of the differents values
        """
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.W_previous: W_previous_,
                                                self.pf_value_previous: pf_value_previous_,
                                                self.dailyReturn_t: dailyReturn_t_})
        # vars = tf.trainable_variables()
        # print(vars) #some infos about variables...
        # vars_vals = self.sess.run(vars)
        # for var, val in zip(vars, vars_vals):
        #     print("var: {}, value: {}".format(var.name, val))
        # print('ok')



class TradeEnv():

    """
    This class is the trading environment (render) of our project. 
    The trading agent calls the class by giving an action at the time t. 
    Then the render gives back the new portfolio at the next step (time t+1). 
    #parameters:
    - windonw_length: this is the number of time slots looked in the past to build the input tensor
    - portfolio_value: this is the initial value of the portfolio 
    - trading_cost: this is the cost (in % of the traded stocks) the agent will pay to execute the action 
    - interest_rate: this is the rate of interest (in % of the money the agent has) the agent will:
        -get at each step if he has a positive amount of money 
        -pay if he has a negative amount of money
    -train_size: % of data taken for the training of the agent - please note the training data are taken with respect 
    of the time span (train -> | time T | -> test)
    """

    def __init__(self, path = './np_data/input.npy', window_length=30,
                 portfolio_value= 10000, trading_cost= 0.25/100,interest_rate= 0.02/250, train_size = 0.7):
        
        #path to numpy data
        self.path = path
        #load the whole data
        self.data = np.load(self.path)


        #parameters
        self.portfolio_value = portfolio_value
        self.window_length=window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        #number of stocks and features
        self.nb_stocks = self.data.shape[1]
        self.nb_features = self.data.shape[0]
        self.end_train = int((self.data.shape[2]-self.window_length)*train_size)
        
        #init state and index
        self.index = None
        self.state = None
        self.done = False

        #init seed
        self.seed()

    def return_pf(self):
        """
        return the value of the portfolio
        """
        return self.portfolio_value
        
    def readTensor(self,X,t):
        ## this is not the tensor of equation 18 
        ## need to batch normalize if you want this one 
        return X[ : , :, t-self.window_length:t ]
    
    def readUpdate(self, t):
        #return the return of each stock for the day t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, w_init, p_init, t=0 ):
        
        """ 
        This function restarts the environment with given initial weights and given value of portfolio
        """
        self.state= (self.readTensor(self.data, self.window_length) , w_init , p_init )
        self.index = t  +self.window_length
        self.done = False
        
        return self.state, self.done

    def step(self, action):
        """
        This function is the main part of the render. 
        At each step t, the trading agent gives as input the action he wants to do. So, he gives the new value of the weights of the portfolio. 
        The function computes the new value of the portfolio at the step (t+1), it returns also the reward associated with the action the agent took. 
        The reward is defined as the evolution of the the value of the portfolio in %. 
        """

        index = self.index
        #get Xt from data:
        data = self.readTensor(self.data, index)
        done = self.done
        
        #beginning of the day 
        state = self.state
        w_previous = state[1]
        pf_previous = state[2]
        

        #the update vector is the vector of the opening price of the day divided by the opening price of the previous day
        update_vector = self.readUpdate(index)

        #allocation choice 
        w_alloc = action
        pf_alloc = pf_previous
        
        #Compute transaction cost
        cost = pf_alloc * np.linalg.norm((w_alloc-w_previous),ord = 1)* self.trading_cost
        
        #convert weight vector into value vector 
        v_alloc = pf_alloc*w_alloc
        
        #pay transaction costs
        pf_trans = pf_alloc - cost
        v_trans = v_alloc - np.array([cost]+ [0]*self.nb_stocks)
        
        #####market prices evolution 
        #we go to the end of the day 
        
        #compute new value vector 
        v_evol = v_trans*update_vector

        
        #compute new portfolio value
        pf_evol = np.sum(v_evol)
        
        #compute weight vector 
        w_evol = v_evol/pf_evol
        w_evol =np.where(w_evol<0,0,w_evol)
        
        #compute instanteanous reward
        reward = (pf_evol-pf_previous)/pf_previous
        
        #update index
        index = index+1
        
        #compute state
        
        state = (data, w_evol, pf_evol) #self.readTensor(self.data, index) having problem 
        
        if index >= self.end_train:
            done = True
        
        self.state = state
        self.index = index
        self.done = done
        
        return state, reward, done

path_data = '/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/data/jiang_data.npy' #Path(__file__).with_name('jiang_input_45Tic.npy')
data = np.load(path_data)
trading_period = data.shape[2]
nb_feature_map = data.shape[0]
nb_stocks = data.shape[1]


# fix parameters of the network
m = nb_stocks

###############################dictionaries of the problem###########################
dict_hp_net = {'n_filter_1': 2, 'n_filter_2': 20, 'kernel1_size':(1, 3)}
dict_hp_pb = {'batch_size': 2**7, 'ratio_train': 0.7,'ratio_val': 0.1, 'length_tensor': 20,
              'ratio_greedy':0.6, 'ratio_regul': 0.01}
dict_hp_opt = {'regularization': 1e-8, 'learning': 9e-2}
dict_fin = {'trading_cost': 0.25/100, 'interest_rate': 0.02/250, 'cash_bias_init': 0.7}
dict_train = {'pf_init_train': 500000, 'w_init_train': 'd', 'n_episodes':20, 'n_batches':5}
dict_test = {'pf_init_test': 500000, 'w_init_test': 'd'}

list_stock =  ['1101','1102','1216','1301','1303','1326','1402','1590','2002','2207',\
 '2303','2308','2317','2327','2330','2357','2379','2382','2395','2408',\
 '2409','2412','2454','2603','2609','2615','2801','2880','2881','2882',\
 '2883','2884','2885','2886','2891','2892','2912','3008l','3034','3045',\
 '3481','4904','4938','6505','9910']
###############################HP of the network ###########################
n_filter_1 = dict_hp_net['n_filter_1']
n_filter_2 = dict_hp_net['n_filter_2']
kernel1_size = dict_hp_net['kernel1_size']

###############################HP of the problem###########################

# Size of mini-batch during training
batch_size = dict_hp_pb['batch_size']
# Total number of steps for pre-training in the training set
total_steps_train = int(dict_hp_pb['ratio_train']*trading_period)

# Total number of steps for pre-training in the validation set
total_steps_val = int(dict_hp_pb['ratio_val']*trading_period)

# Total number of steps for the test
total_steps_test = trading_period-total_steps_train-total_steps_val

# Number of the columns (number of the trading periods) in each input price matrix
n = dict_hp_pb['length_tensor']

ratio_greedy = dict_hp_pb['ratio_greedy']

ratio_regul = dict_hp_pb['ratio_regul']

##############################HP of the optimization###########################


# The L2 regularization coefficient applied to network training
regularization = dict_hp_opt['regularization']
# Parameter alpha (i.e. the step size) of the Adam optimization
learning = dict_hp_opt['learning']

optimizer =tf.train.AdamOptimizer(learning)
# optimizer = tf.optimizers.Adam(learning) # 


##############################Finance parameters###########################

trading_cost= dict_fin['trading_cost']
interest_rate= dict_fin['interest_rate']
cash_bias_init = dict_fin['cash_bias_init']

############################## PVM Parameters ###########################
sample_bias = 5e-5  # Beta in the geometric distribution for online training sample batches


############################## Training Parameters ###########################

w_init_train = np.array(np.array([1.0]+[.0]*m))#dict_train['w_init_train']

pf_init_train = dict_train['pf_init_train']

n_episodes = dict_train['n_episodes']
n_batches = dict_train['n_batches']

############################## Test Parameters ###########################

w_init_test = np.array(np.array([1.0]+[0]*m))#dict_test['w_init_test']

pf_init_test = dict_test['pf_init_test']


############################## other environment Parameters ###########################

w_eq = np.array(np.array([1/(m+1)]*(m+1)))

w_s = np.array(np.array([1]+[0.0]*m))


# random action function

def get_random_action(m):
    random_vec = np.random.rand(m+1)
    return random_vec/np.sum(random_vec)

#environment for trading of the agent 
# this is the agent trading environment (policy network agent)
env = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])


#environment for equiweighted
#this environment is set up for an agent who only plays an equiweithed portfolio (baseline)
env_eq = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])

#environment secured (only money)
#this environment is set up for an agentwho plays secure, keeps its money
env_s = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])

#full on one stock environment 
#these environments are set up for agents who play only on one stock

action_fu = list()
env_fu = list()


for i in range(m):
    action = np.array([0]*(i+1) + [1] + [0]*(m-(i+1)))
    action_fu.append(action)
    
    env_fu_i = TradeEnv(path=path_data, window_length=n,
               portfolio_value=pf_init_train, trading_cost=trading_cost,
               interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])
    
    env_fu.append(env_fu_i)

# define neural net \pi_\phi(s) as a class
class Policy(object):
    '''
    This class is used to instanciate the policy network agent

    '''

    def __init__(self, m, n, sess, optimizer,
                 trading_cost=trading_cost,
                 interest_rate=interest_rate,
                 n_filter_1=n_filter_1,
                 n_filter_2=n_filter_2):

        # parameters
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.n = n
        self.m = m

        with tf.variable_scope("Inputs"):

            # Placeholder

            # tensor of the prices
            self.X_t = tf.placeholder(
                tf.float32, [None, nb_feature_map, self.m, self.n])  # The Price tensor
            # weights at the previous time step
            self.W_previous = tf.placeholder(tf.float32, [None, self.m+1])
            # portfolio value at the previous time step
            self.pf_value_previous = tf.placeholder(tf.float32, [None, 1])
            # vector of Open(t+1)/Open(t)
            self.dailyReturn_t = tf.placeholder(tf.float32, [None, self.m])
            
            #self.pf_value_previous_eq = tf.placeholder(tf.float32, [None, 1])
            
            

        with tf.variable_scope("Policy_Model"):

            # variable of the cash bias
            bias = tf.get_variable('cash_bias', shape=[
                                   1, 1, 1, 1], initializer=tf.constant_initializer(cash_bias_init))
            # shape of the tensor == batchsize
            shape_X_t = tf.shape(self.X_t)[0]
            # trick to get a "tensor size" for the cash bias
            self.cash_bias = tf.tile(bias, tf.stack([shape_X_t, 1, 1, 1]))
            # print(self.cash_bias.shape)

            with tf.variable_scope("Conv1"):
                # first layer on the X_t tensor
                # return a tensor of depth 2
                self.conv1 = tf.layers.conv2d(
                    inputs=tf.transpose(self.X_t, perm=[0, 3, 2, 1]),
                    activation=tf.nn.relu,
                    filters=self.n_filter_1,
                    strides=(1, 1),
                    kernel_size=kernel1_size,
                    padding='same')

            with tf.variable_scope("Conv2"):
                
                #feature maps
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1,
                    activation=tf.nn.relu,
                    filters=self.n_filter_2,
                    strides=(self.n, 1),
                    kernel_size=(1, self.n),
                    padding='same')

            with tf.variable_scope("Tensor3"):
                #w from last periods
                # trick to have good dimensions
                w_wo_c = self.W_previous[:, 1:]
                w_wo_c = tf.expand_dims(w_wo_c, 1)
                w_wo_c = tf.expand_dims(w_wo_c, -1)
                self.tensor3 = tf.concat([self.conv2, w_wo_c], axis=3)

            with tf.variable_scope("Conv3"):
                #last feature map WITHOUT cash bias
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2,
                    activation=tf.nn.relu,
                    filters=1,
                    strides=(self.n_filter_2 + 1, 1),
                    kernel_size=(1, 1),
                    padding='same')

            with tf.variable_scope("Tensor4"):
                #last feature map WITH cash bias
                self.tensor4 = tf.concat([self.cash_bias, self.conv3], axis=2)
                # we squeeze to reduce and get the good dimension
                self.squeezed_tensor4 = tf.squeeze(self.tensor4, [1, 3])

            with tf.variable_scope("Policy_Output"):
                # softmax layer to obtain weights
                self.action = tf.nn.softmax(self.squeezed_tensor4)
                #tf.nn.softmax(self.squeezed_tensor4)
                
            with tf.variable_scope("Reward"):
                # computation of the reward
                # please look at the chronological map to understand
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
                Vprime_t = self.action * self.pf_value_previous
                Vprevious = self.W_previous*self.pf_value_previous

                # this is just a trick to get the good shape for cost
                constant = tf.constant(1.0, shape=[1])

                cost = self.trading_cost * \
                    tf.norm(Vprime_t-Vprevious, ord=1, axis=1)*constant

                cost = tf.expand_dims(cost, 1)

                zero = tf.constant(
                    np.array([0.0]*m).reshape(1, m), shape=[1, m], dtype=tf.float32)

                vec_zero = tf.tile(zero, tf.stack([shape_X_t, 1]))
                vec_cost = tf.concat([cost, vec_zero], axis=1)

                Vsecond_t = Vprime_t - vec_cost

                V_t = tf.multiply(Vsecond_t, y_t)
                self.portfolioValue = tf.norm(V_t, ord=1)
                self.instantaneous_reward = (
                    self.portfolioValue-self.pf_value_previous)/self.pf_value_previous
                
                
            with tf.variable_scope("Reward_Equiweighted"):
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
  

                V_eq = w_eq*self.pf_value_previous
                V_eq_second = tf.multiply(V_eq, y_t)
        
                self.portfolioValue_eq = tf.norm(V_eq_second, ord=1)
            
                self.instantaneous_reward_eq = (
                    self.portfolioValue_eq-self.pf_value_previous)/self.pf_value_previous
                
            with tf.variable_scope("Max_weight"):
                self.max_weight = tf.reduce_max(self.action)
                print(self.max_weight.shape)

                
            with tf.variable_scope("Reward_adjusted"):
                
                self.adjested_reward = self.instantaneous_reward - self.instantaneous_reward_eq - ratio_regul*self.max_weight
                
        # objective function 
        # maximize reward over the batch 
        # min(-r) = max(r)
        self.train_op = optimizer.minimize(self.adjested_reward)
        
        # some bookkeeping
        self.optimizer = optimizer
        self.sess = sess
        tf.summary.FileWriter("Jiang_replicate/logs/", self.sess.graph)

    def compute_W(self, X_t_, W_previous_):
        """
        This function returns the action the agent takes 
        given the input tensor and the W_previous
        
        It is a vector of weight

        """

        return self.sess.run(tf.squeeze(self.action), feed_dict={self.X_t: X_t_, self.W_previous: W_previous_})

    def train(self, X_t_, W_previous_, pf_value_previous_, dailyReturn_t_):
        """
        This function trains the neural network
        maximizing the reward 
        the input is a batch of the differents values
        """
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.W_previous: W_previous_,
                                                self.pf_value_previous: pf_value_previous_,
                                                self.dailyReturn_t: dailyReturn_t_})
        # vars = tf.trainable_variables()
        # print(vars) #some infos about variables...
        # vars_vals = self.sess.run(vars)
        # for var, val in zip(vars, vars_vals):
        #     print("var: {}, value: {}".format(var.name, val))
        # print('ok')

class PVM(object):
    '''
    This is the memory stack called PVM in the paper
    '''

    def __init__(self, m, sample_bias, total_steps = total_steps_train, 
                 batch_size = batch_size, w_init = w_init_train):
        
        
        #initialization of the memory 
        #we have a total_step_times the initialization portfolio tensor 
        self.memory = np.transpose(np.array([w_init]*total_steps))  
        self.sample_bias = sample_bias
        self.total_steps = total_steps
        self.batch_size = batch_size

    def get_W(self, t):
        #return the weight from the PVM at time t 
        return self.memory[:, t]

    def update(self, t, w):
        #update the weight at time t
        self.memory[:, t] = w
        


    def draw(self, beta=sample_bias):
        '''
        returns a valid step so you can get a training batch starting at this step
        '''
        while 1:
            z = np.random.geometric(p=beta)
            tb = self.total_steps - self.batch_size + 1 - z
            if tb >= 0:
                return tb
            
    def test(self):
        #just to test
        return self.memory


def get_max_draw_down(xs):
    xs = np.array(xs)
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
    
    return xs[j] - xs[i]

def eval_perf(e,actor):
    """
    This function evaluates the performance of the different types of agents. 
    
    
    """
    list_weight_end_val = list()
    list_pf_end_training = list()
    list_pf_min_training = list()
    list_pf_max_training = list()
    list_pf_mean_training = list()
    list_pf_dd_training = list()
    list_reward = list()
    
    #######TEST#######
    #environment for trading of the agent 
    env_eval = TradeEnv(path=path_data, window_length=n,
                   portfolio_value=pf_init_train, trading_cost=trading_cost,
                   interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])



    #initialization of the environment 
    state_eval, done_eval = env_eval.reset(w_init_test, pf_init_test, t = total_steps_train)



    #first element of the weight and portfolio value 
    p_list_eval = [pf_init_test]
    w_list_eval = [w_init_test]
    reward_list = [0]

    for k in range(total_steps_train, total_steps_train +total_steps_val-int(n/2)):
        X_t = state_eval[0].reshape([-1]+ list(state_eval[0].shape))
        W_previous = state_eval[1].reshape([-1]+ list(state_eval[1].shape))
        pf_value_previous = state_eval[2]
        #compute the action 
        action = actor.compute_W(X_t, W_previous)
        #step forward environment 
        state_eval, reward_eval, done_eval = env_eval.step(action)

        X_next = state_eval[0]
        W_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        dailyReturn_t = X_next[-1, :, -1]
        #print('current portfolio value', round(pf_value_previous,0))
        #print('weights', W_previous)
        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(W_t_eval)
        reward_list.append(reward_eval)
    
    mean_reward = np.mean(np.array(reward_list))
    list_weight_end_val.append(w_list_eval[-1])
    list_pf_end_training.append(p_list_eval[-1])
    list_pf_min_training.append(np.min(p_list_eval))
    list_pf_max_training.append(np.max(p_list_eval))
    list_pf_mean_training.append(np.mean(p_list_eval))
    
    list_pf_dd_training.append(get_max_draw_down(p_list_eval))

    print('End of test PF value:',round(p_list_eval[-1]))
    print('Min of test PF value:',round(np.min(p_list_eval)))
    print('Max of test PF value:',round(np.max(p_list_eval)))
    print('Mean of test PF value:',round(np.mean(p_list_eval)))
    print('Max Draw Down of test PF value:',round(get_max_draw_down(p_list_eval)))
    print('End of test weights:',w_list_eval[-1])
    print('Average Reward',mean_reward)
    plt.title('Portfolio evolution (validation set) episode {}'.format(e))
    plt.plot(p_list_eval, label = 'Agent Portfolio Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()
    plt.title('Portfolio weights (end of validation set) episode {}'.format(e))
    plt.bar(np.arange(m+1), list_weight_end_val[-1])
    plt.xticks(np.arange(m+1), ['Money'] + list_stock, rotation=45)
    #plt.show()
    
    
    names = ['Money'] + list_stock
    w_list_eval = np.array(w_list_eval)
    # for j in range(m+1):
    #     plt.plot(w_list_eval[:,j], label = 'Weight Stock {}.tw'.format(names[j]))
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
    # plt.show()
    
    return mean_reward

weightPath = '/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/jiang/trial/my_test_model-1000.meta-1000.meta-1000.meta'
tf.reset_default_graph()
_ = tf.train.import_meta_graph(weightPath)
sess3 = tf.Session()
actor = Policy(m, n, sess3, optimizer,trading_cost=trading_cost,interest_rate=interest_rate) 
init_op = tf.initialize_all_variables()

actor.sess.run(init_op)

print(actor)
print(actor)

state_fu = [0]*m
done_fu = [0]*m

#########################################TEST##########################################

#initialization of the environment 
state, done = env.reset(w_init_test, pf_init_test, t = total_steps_train)

state_eq, done_eq = env_eq.reset(w_eq, pf_init_test, t = total_steps_train)
state_s, done_s = env_s.reset(w_s, pf_init_test, t = total_steps_train)

for i in range(m):
    state_fu[i],  done_fu[i] = env_fu[i].reset(action_fu[i], pf_init_test, t = total_steps_train)


#first element of the weight and portfolio value 
p_list = [pf_init_test]
w_list = [w_init_test]

p_list_eq = [pf_init_test]
p_list_s = [pf_init_test]


p_list_fu = list()
for i in range(m):
    p_list_fu.append([pf_init_test])
    
pf_value_t_fu = [.0]*m
    
index_list = [1827]
list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t = [], [], [], []
action_mem = []
for k in range(1827, 2690):
    index_list.append(k)
    X_t = state[0].reshape([-1]+ list(state[0].shape))
    W_previous = state[1].reshape([-1]+ list(state[1].shape))
    pf_value_previous = state[2]
    #compute the action 
    action = actor.compute_W(X_t, W_previous)
    #step forward environment 
    state, reward, done = env.step(action)
    state_eq, reward_eq, done_eq = env_eq.step(w_eq)
    state_s, reward_s, done_s = env_s.step(w_s)
    
    
    for i in range(m):
        state_fu[i], _ , done_fu[i] = env_fu[i].step(action_fu[i])
    
    
    X_next = state[0]
    W_t = state[1]
    pf_value_t = state[2]
    
    pf_value_t_eq = state_eq[2]
    pf_value_t_s = state_s[2]
    for i in range(m):
        pf_value_t_fu[i] = state_fu[i][2]
    
    dailyReturn_t = X_next[-1, :, -1]
    if k%20 == 0:
        print('current portfolio value', round(pf_value_previous,0))
        print('weights', W_previous)
    p_list.append(pf_value_t)
    w_list.append(W_t)
    
    p_list_eq.append(pf_value_t_eq)
    p_list_s.append(pf_value_t_s)
    for i in range(m):
        p_list_fu[i].append(pf_value_t_fu[i])
    
    ###
    # list_X_t.append(X_t.reshape(state[0].shape))
    # list_W_previous.append(W_previous.reshape(state[1].shape))
    # list_pf_value_previous.append([pf_value_previous])
    # list_dailyReturn_t.append(dailyReturn_t)
    # list_X_t = np.array(list_X_t)
    # list_W_previous = np.array(list_W_previous)
    # list_pf_value_previous = np.array(list_pf_value_previous)
    # list_dailyReturn_t = np.array(list_dailyReturn_t)


    # #for each batch, train the network to maximize the reward
    # actor.train(list_X_t, list_W_previous,
    #             list_pf_value_previous, list_dailyReturn_t)
    ###
    #here to breack the loop/not in original code     
    if k== 2040:#total_steps_train +total_steps_val-int(n/2) + 100:
        print('check')
        pass
jiang_data = pd.read_pickle('/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/RLEPR/data/latest_45tic_priceBook.pkl') #('/Users/yi-hsuanlee/Desktop/WIDM/Thesis/交接的Code和Data/Data/1.5_preprocess_data/new_2021/for_jiang_45tic_2012_2021_data.pkl')
jiang_data = jiang_data.loc['2012-01-02':]

from datetime import datetime
dt = datetime.now()
ts = datetime.timestamp(dt)

jiang_portfolio=pd.DataFrame(w_list,columns=np.insert(jiang_data.Stock_ID.unique(),0,0) , index = jiang_data[jiang_data.Stock_ID == 2330][1:].iloc[index_list].index)
jiang_portfolio['Asset']=p_list
jiang_portfolio.to_pickle(f'/Users/yi-hsuanlee/Desktop/WIDM/Thesis/finRL-elegant/dashboard/jiang/latest_jiang_weights_{int(ts)}.pkl')

