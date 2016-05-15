import random
import math
import time
#import threading
from multiprocessing import Pool
#from scipy.optimize import brute
from sklearn.grid_search import ParameterGrid
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
from params import params, valid_params

class Entry:
    def __init__(self,state):
        self.state = tuple(state.values())
    def __hash__(self):
        return hash(self.state)
    def __eq__(self,other):
        return self.state == other.state
    def __ne__(self,other):
        return not(self == other)
    def __repr__(self):
        return repr(self.state)
    def __str__(self):
        return repr(self)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.params = env.params
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here

        # PARAMS
        self.qtable = defaultdict(lambda:[0.0, 0.0, 0.0, 0.0]) 
        # == lambda : [0 for _ in self.valid_actions]

        #epsilon for e-greedy
        self.eps_start = params['eps_start']
        self.eps_cur = self.eps_start
        self.eps_end = params['eps_end']
        self.eps_decay = params['eps_decay'] # -- parameter used when max_epoch is not known 

        #alpha for q-learning rate
        self.alpha_start = params['alpha_start'] #q-learning rate
        self.alpha_cur= self.eps_start
        self.alpha_end = params['alpha_end']
        self.alpha_decay = params['alpha_decay'] # -- parameter used when max_epoch is not known 
        #currently replaced with function

        self.gamma = self.params['gamma'] #temporal discount
        self.valid_actions = self.env.valid_actions

        self.epoch = 0
        self.max_epoch = params['max_epoch']
        
        # STATE
        self.state = None
        self.prev = None

        # Logging
        self.verbose = False
        self.floss = open('loss.csv','w+')

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.prev = None
        self.epoch = self.epoch + 1
        self.floss.flush()


    def update(self, t):
        if self.verbose:
            print '-------------------------------'
        # Gather inputs

        deadline = self.env.get_deadline(self)

        if self.state is None:
            self.state = self.env.sense(self)
            self.next_waypoint = self.planner.next_waypoint()
            self.state['waypoint'] = self.next_waypoint 

        # TODO: Select action according to your policy
        action = self.getAction()

        if self.verbose:
            print 'State', self.state
            print 'Action', self.valid_actions[action]
            print 'TARGET', self.state['waypoint']

        self.prev = self.state
        # Execute action and get reward
        reward = self.env.act(self, self.valid_actions[action])

        # observe new state
        self.state = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        self.state['waypoint'] = self.next_waypoint # from route planner, also displayed by simulator

        # Implementing Q-Learning
        entry = Entry(self.prev)
        #print 'entry', entry
        #print 'maxQ', self.maxQ(self.state)


        q_old = self.qtable[entry][action]
        q_new = reward + self.gamma * self.maxQ(self.state)

        # Logging
        if self.verbose:
            print 'Reward', reward
            print 'q_old', self.qtable[entry][action]
            print 'q_new', q_new

        self.floss.write(str(0.5 * (q_new - q_old)**2) + '\n')
        self.qtable[entry][action] = (1.0 - self.alpha()) * q_old + self.alpha() * q_new

        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state, action, reward)  # [debug]
        #print "Q-Value : {}".format(self.qtable[entry])
        #print "# of total visited states : {}".format(len(self.qtable)) # sanity check
        #print "epsilon : {}".format(self.eps())
        #print "alpha : {}".format(self.alpha())

    def getRand(self):
        return random.randrange(4)
        #return random.choice(self.valid_actions)

    def getBest(self):
        maxVal = -99999 #arbitrary small value
        maxAct = None

        for act,val in enumerate(self.qtable[Entry(self.state)]):
            if maxVal < val:
                maxAct = act
                maxVal = val

        #if self.verbose:
        #    print 'State', self.state
        #    print 'Q' , self.qtable[Entry(self.state)]
        return maxAct

    def getAction(self):
        if random.random() < self.eps():
            return self.getRand()
        else:
            return self.getBest()

    def maxQ(self,state):
        return max(self.qtable[Entry(state)])

    def eps(self):
        if self.params['eps_anneal'] == 'tanh': #curvy anneal
            return self.eps_start * (1.0 - math.tanh(2*float(self.epoch)/self.max_epoch))
        elif self.params['eps_anneal'] == 'linear': #linear anneal
            return self.eps_start + (self.eps_end - self.eps_start) * self.epoch / self.max_epoch
        elif self.params['eps_anneal'] == 'decay': #exponential anneal
            self.eps_cur *= self.eps_decay
            return max(self.eps_end, self.eps_cur)
        else:
            return self.eps_cur

    def alpha(self):
        if self.params['alpha_anneal'] == 'tanh':
            #uses alpha_start
            return self.alpha_start * (1.0 - math.tanh(2*float(self.epoch)/self.max_epoch))
        elif self.params['alpha_anneal'] == 'linear':
            #uses alpha_start, alpha_end
            return self.alpha_start + (self.alpha_end - self.alpha_start) * self.epoch / self.max_epoch
        elif self.params['alpha_anneal'] == 'decay':
            #uses alpha_start, alpha_decay
            self.alpha_cur *= self.alpha_decay
            return max(self.alpha_end, self.alpha_cur)
        else:
            #uses alpha_start
            return self.alpha_cur

        



def run(params):
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment(params)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0,silent=False)  # reduce update_delay to speed up simulation
    sim.run(n_trials=params['max_epoch'])  # press Esc or close pygame window to quit
    return params, sim.score()

def run_silent(params):
    """Run the agent for a finite number of trials."""
    #print 'run_silent({})'.format(params)
    # Set up environment and agent
    e = Environment(params)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0,silent=True)  # reduce update_delay to speed up simulation
    sim.run(n_trials=params['max_epoch'])  # press Esc or close pygame window to quit
    return params, sim.run(n_trials=params['max_epoch'])

def gridSearch(valid_params):
    grid = ParameterGrid(valid_params)
    best_params = None
    best_score = -99999

    total = len(grid)
    start = time.time()
    last = time.time()

    for index, params in enumerate(grid):
        print '{}/{}'.format(index,total)
        try:
            #score = run(params)
            score = run_silent(params)
            if score > best_score:
                best_params = params
                best_score = score
        except Exception as inst:
            print 'SWALLOWING EXCEPTION : CANT RISK ABORTING', inst

        now = time.time()
        print '{} seconds spent'.format(now - last)
        last = now

    print 'best params', best_params
    print 'best score', best_score

def gridSearch_2(valid_params):
    pool = Pool(processes=4)              # start 4 worker processes
    grid = ParameterGrid(valid_params)
    res = pool.map(run_silent, grid)
    
    maxParam = None
    maxScore = -99999.0

    print res
    for param,score in res:
        if score > maxScore:
            maxScore = score
            maxParam = param

    print 'max Score', maxScore
    print 'max Param', maxParam

if __name__ == '__main__':
    print run_silent(params) #-- run once 
    #gridSearch(valid_params)
    #gridSearch_2(valid_params)
