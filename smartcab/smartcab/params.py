params = {
    'max_epoch' : 1000, #??
    'eps_start' : 1.0, #no real need to change
    'eps_end' : 0.0, #some need to change
    'eps_decay' : 0.995, #some need to change
    'eps_anneal' : 'linear', #?
    'alpha_start' : 0.9, #great need to change
    'alpha_end' : 0.1,
    'alpha_decay' : 0.99,
    'alpha_anneal' : 'tanh', #great need to change
    'gamma' : 0.99, #great need to change
    'reward_none' : -0.1,
    'reward_bonus' : 10,
    'reward_correct' : 2,
    'reward_legal' : 0.5,
    'reward_illegal' : -1,
}


valid_params = {
    'max_epoch' : (100,), #don't want it longer
    'eps_start' : (1.0,),
    'eps_end' : (0.1,0.0),
    'eps_decay' : (0.99,), #cutting variation
    'eps_anneal' : ('linear',None), #cutting tanh, decay
    'alpha_start' : (0.9,0.6),
    'alpha_end' : (0.05,), #cutting variation
    'alpha_decay' : (0.99,), #cutting variation
    'alpha_anneal' : ('tanh',None), #cutting linear, decay
    'gamma' : (0.99,0.9), #cutting variation
    'reward_none' : (1.0,-0.1),
    'reward_bonus' : (10.0,1.0),
    'reward_correct' : (5.0,2.0),
    'reward_legal' : (1.0,0.5),
    'reward_illegal' : (-1.0,) #cutting variation
}
