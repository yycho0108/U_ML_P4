
params = {
    'save': False,
    'silent': True,
    'max_epoch': 100,
    'eps_start': 0.25,
    'eps_end': 0.05,
    'eps_decay': 0.99,
    'eps_anneal': 'linear',
    'alpha_start': 0.35,
    'alpha_end': 0.05,
    'alpha_decay': 0.99,
    'alpha_anneal': 'linear',
    'gamma': 0.05,
    'reward_none': 1.0,
    'reward_bonus': 10.0,
    'reward_correct': 2.0,
    'reward_legal': 0.5,
    'reward_illegal': -1.0,
}



#grid-search #1
#search_params = {
#    'max_epoch' : (100,), #don't want it longer
#    'eps_start' : (1.0,),
#    'eps_end' : (0.0,0.1), --> 0.1
#    'eps_decay' : (0.99,), #cutting variation
#    'eps_anneal' : ('linear',None), #cutting tanh, decay --> None
#    'alpha_start' : (0.6,0.9), --> 0.9
#    'alpha_end' : (0.05,), #cutting variation
#    'alpha_decay' : (0.99,), #cutting variation
#    'alpha_anneal' : ('tanh',None), #cutting linear, decay --> None
#    'gamma' : (0.9,0.99), #cutting variation --> 0.9
#    'reward_none' : (-0.1,1.0), --> -0.1
#    'reward_bonus' : (1.0,10.0), --> 10.0
#    'reward_correct' : (2.0,5.0), --> 5.0
#    'reward_legal' : (0.5,1.0), --> 0.5
#    'reward_illegal' : (-1.0,) #cutting variation
#}

#grid-search #2
#search_params= {
#    'max_epoch' : (100,), #don't want it longer
#    'eps_start' : (1.0,),
#    'eps_end' : (0.15,0.1,0.05),
#    'eps_decay' : (0.99,), #cutting variation
#    'eps_anneal' : ('linear',None), #cutting tanh, decay
#    'alpha_start' : (0.8,0.9,0.95),
#    'alpha_end' : (0.05,), #cutting variation
#    'alpha_decay' : (0.99,), #cutting variation
#    'alpha_anneal' : ('tanh',None), #cutting linear, decay
#    'gamma' : (0.8,0.9,0.95), #cutting variation
#    'reward_none' : (-0.1,-0.5),
#    'reward_bonus' : (2.0,5.0,10.0),
#    'reward_correct' : (3.0,7.0),
#    'reward_legal' : (0.25,0.75),
#    'reward_illegal' : (-1.5,-1.0,-0.5) #cutting variation
#}

#grid-search #3
#search_params= {
#    'max_epoch' : (100,), #don't want it longer
#    'eps_start' : (1.0,),
#    'eps_end' : (0.05,),
#    'eps_decay' : (0.99,), #cutting variation
#    'eps_anneal' : ('linear',None), #cutting tanh, decay
#    'alpha_start' : (0.1,0.3,0.6),
#    'alpha_end' : (0.0,0.05,), #cutting variation
#    'alpha_decay' : (0.99,), #cutting variation
#    'alpha_anneal' : ('tanh','linear',None), #cutting linear, decay
#    'gamma' : (0.8,0.9,0.95), #cutting variation
#    'reward_none' : (1.0,),
#    'reward_bonus' : (10.0,),
#    'reward_correct' : (2.0,),
#    'reward_legal' : (0.5,),
#    'reward_illegal' : (-1.0,)
#}
#grid-search #4 - searching for alpha
#search_params= {
#    'max_epoch' : (100,), #don't want it longer
#    'eps_start' : (1.0,),
#    'eps_end' : (0.05,),
#    'eps_decay' : (0.99,),
#    'eps_anneal' : ('linear'),
#    'alpha_start' : (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
#    'alpha_end' : (0.05,),
#    'alpha_decay' : (0.99,),
#    'alpha_anneal' : ('tanh'),
#    'gamma' : (0.8,),
#    'reward_none' : (1.0,),
#    'reward_bonus' : (10.0,),
#    'reward_correct' : (2.0,),
#    'reward_legal' : (0.5,),
#    'reward_illegal' : (-1.0,)
#}

#grid-search #4 - searching for alpha-gamma combination
#search_params= {
#    'silent' : (True,),
#    'save' : (False,),
#    'max_epoch' : (100,), #don't want it longer
#    'eps_start' : (0.1,0.3,0.5,0.7,0.9),
#    'eps_end' : (0.05,),
#    'eps_decay' : (0.99,),
#    'eps_anneal' : ('linear',None),
#    'alpha_start' : (0.1,0.3,0.5,0.7,0.9),
#    'alpha_end' : (0.05,),
#    'alpha_decay' : (0.99,),
#    'alpha_anneal' : ('linear',None),
#    'gamma' : (0.1,0.3,0.5,0.7,0.9),
#    'reward_none' : (1.0,),
#    'reward_bonus' : (10.0,),
#    'reward_correct' : (2.0,),
#    'reward_legal' : (0.5,),
#    'reward_illegal' : (-1.0,)
#}

#grid-search #5 - searching for best reward combination
#search_params = {
#        'max_epoch' : (100,),
#        'eps_start' : (0.1,),
#        'eps_end' : (0.05,),
#        'eps_decay' : (0.995,),
#        'eps_anneal' : ('linear',),
#        'alpha_start' : (0.9,),
#        'alpha_end' : (0.05,),
#        'alpha_decay' : (0.99,),
#        'alpha_anneal' : (None,),
#        'gamma' : (0.3,),
#        'reward_none' : (-0.1,0.0,1.0,),
#        'reward_bonus' : (0.0,5.0,10.0,),
#        'reward_correct' : (1.,2.,5.,),
#        'reward_legal' : (0.5,1.,2.),
#        'reward_illegal' : (-1.,-2.,-5.,),
#}

#grid-search #6 - a-b-c combination
search_params = {
    'save': (False,),
    'silent': (True,),
    'max_epoch': (100,),
    'eps_start': (0.25,),
    'eps_end': (0.05,),
    'eps_decay': (0.99,),
    'eps_anneal': ('linear',),
    'alpha_start': (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    'alpha_end': (0.05,),
    'alpha_decay': (0.99,),
    'alpha_anneal': ('linear',),
    'gamma': (0.05,),
    'reward_none': (1.0,),
    'reward_bonus': (10.0,),
    'reward_correct': (2.0,),
    'reward_legal': (0.5,),
    'reward_illegal': (-1.0,),
}
