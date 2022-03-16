
##### DIM OPTIMALITY IS A VERY HACKY SCRIPT AND DEPENDS ON THE STRUCTURE OF DIM_REDUCE_JOB.PY

##### IF YOU'RE INTERESTED IN USING THIS, CONTACT ME; SEE THE CODEBASE DESCRIPTION FOR WHY MAY NOT
##### BE A USEFUL IDEA FOR A SCRIPT ANYWAY

dim_op_params = {
    'm_min': 1,  # smallest m to test
    'm_max': 3,  # largest m to test
    'by': 1,  # gap between m's to test
    'greedy': True  # use optimal transformation in last dimension to initialize run in next dimension?
}


