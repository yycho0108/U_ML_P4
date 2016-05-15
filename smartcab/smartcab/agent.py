import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict

class Entry:
    def __init__(self,state,action):
        self.state = tuple(state.values())
        self.action = action
    def __hash__(self):
        return hash((self.state,self.action))
    def __eq__(self,other):
        return (self.state,self.action) == (other.state,other.action)
    def __ne__(self,other):
        return not(self == other)
    def __repr__(self):
        return repr({'state' : self.state, 'action' : self.action})
    def __str__(self):
        return repr(self)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here

        # PARAMS
        self.qtable = defaultdict(lambda:10.0) 

        #epsilon for e-greedy
        self.eps_start = 1.0
        self.eps_cur = self.eps_start
        self.eps_end = 0.0
        self.eps_decay = 0.999 

        self.alpha = 0.5 #q-learning rate
        self.gamma = 0.8 #temporal discount
        self.valid_actions = self.env.valid_actions
        self.epoch = 0
        
        # STATE
        self.state = None
        self.prev = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.prev = None


    def update(self, t):
        # Gather inputs

        deadline = self.env.get_deadline(self)

        if self.state is None:
            self.state = self.env.sense(self)
            self.state['waypoint'] = self.planner.next_waypoint()

        # TODO: Select action according to your policy
        action = self.getAction()

        self.prev = self.state
        # Execute action and get reward
        reward = self.env.act(self, action)

        # observe new state
        self.state = self.env.sense(self)
        self.state['waypoint'] = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        # Implementing Q-Learning
        entry = Entry(self.prev,action)
        #print 'entry', entry
        print 'maxQ', self.maxQ(self.state)
        self.qtable[entry] = (1.0 - self.alpha) * self.qtable[entry] \
                + (self.alpha) * (reward + self.gamma * self.maxQ(self.state)) 

        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state, action, reward)  # [debug]
        #print "Q-Value : {}".format(self.qtable[entry])
        #print "# of total visited states : {}".format(len(self.qtable)) # sanity check
        #print "epsilon : {}".format(self.eps())

    def getRand(self):
        return random.choice(self.valid_actions)

    def getBest(self):
        maxVal = -99999 #arbitrary small value
        maxAct = None

        for action in self.valid_actions:
            val = self.qtable[Entry(self.state,action)]
            if maxVal < val:
                maxAct = action
                maxVal = val

        return maxAct

    def getAction(self):
        if random.random() < self.eps():
            return self.getRand()
        else:
            return self.getBest()

    def maxQ(self,state):
        return max([self.qtable[Entry(state,action)] for action in self.valid_actions])

    def eps(self):
        self.eps_cur *= self.eps_decay
        return max(self.eps_end, self.eps_cur)
        


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
