# script for comparing models on holdout data by the computation of average nat advantage per turn

import sys
sys.path.append('.')
import numpy as np


jobctl = {# some settings (too simple a script for a job file)

"seed" : 1,            # Int or None - Seed for tf graph operations and all tf random initialization

'model_1': './Figures/Wind_Exp41',  # models to compare
'model_2': './Figures/Wind_Exp46',

}

globals().update(jobctl)
np.random.seed(seed)

# load in the saved likelihoods from running the sharpness script
conditional_lik1 = np.load(model_1 + '/' + 'log_test_liks.npy')
conditional_lik2 = np.load(model_2 + '/' + 'log_test_liks.npy')

print("Difference in average conditional log-likelihood over the "
      "test set: {}".format(np.mean(conditional_lik1 - conditional_lik2)))
