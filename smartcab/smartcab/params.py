params = {
    'max_epoch' : 100, #??
    'eps_start' : 1.0, #no real need to change
    'eps_end' : 0.0, #some need to change
    'eps_decay' : 0.995, #some need to change
    'eps_anneal' : 'linear', #?
    'alpha_start' : 0.8, #great need to change
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
