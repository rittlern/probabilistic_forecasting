import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_functions import get_minibatch, path_finder


from dim_reduce_job import training_params, num_training_phases
from generate_data_job import data_params
from read_data_job import read_data_params

############################################ SETUP ####################################################
sys.path.insert(0, ".")

# suppress depreciation warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# parameter setup. no training phases

globals().update(training_params)

if synthetic_data:
    globals().update(data_params)
else:
    globals().update(read_data_params)

# find the correct paths for loading and saving
stem = 'Experiments/' if synthetic_data else 'Real_Experiments/'
path, save_path = path_finder(stem, data_directory)

x_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
x_validation = np.load(path + 'x_validation.npy')
y_validation = np.load(path + 'y_validation.npy')

idx1 = np.random.choice(np.array(list(range(0, x_validation.shape[0]))), size=x_validation.shape[0]/2, replace=False)
xv1 = x_validation[idx1, :]
yv1 = y_validation[:, idx1]
idx2 = [i for i in range(x_validation.shape[0]) if i not in idx1]
xv2 = x_validation[idx2, :]

y_dim = y_train.shape[0]
d = x_train.shape[1]



################################################ TRAINING ##################################################
# define computation graph
np.random.seed(training_seed)
validation_losses = list()
batch_training_losses = list()
learning_rate = .0001
batch_size = 100
num_steps = 100000
### training
tf.reset_default_graph()
X1 = tf.placeholder(tf.float64, shape=(d, batch_size), name='X1')  # need two sets of samples to get independent draws from marginals
X2 = tf.placeholder(tf.float64, shape=(d, batch_size), name='X2')
Y1 = tf.placeholder(tf.float64, shape=(y_dim, batch_size), name="Y1")  # need two sets of samples to get independent draws from marginals
X1v = tf.Variable(np.transpose(xv1), name="Xv", trainable=False)
X2v = tf.Variable(np.transpose(xv2), name="Xv", trainable=False)
Y1v = tf.Variable(yv1, name="Yv", trainable=False)


T = tf.Variable(np.random.normal(0, 1, size=[m, d]), name="T", trainable=True)
W1 = tf.Variable(np.random.normal(0, 1, size=[m+y_dim, m+y_dim]), name="W1", trainable=True)
b1 = tf.Variable(np.random.normal(0, 1, size=[m+y_dim, 1]), name="b1", trainable=True)
W2 = tf.Variable(np.random.normal(0, 1, size=[m+y_dim, m+y_dim]), name="W2", trainable=True)
b2 = tf.Variable(np.random.normal(0, 1, size=[m+y_dim, 1]), name="b2", trainable=True)
Wf = tf.Variable(np.random.normal(0, 1, size=[1, m+y_dim]), name="Wf", trainable=True)
trainable_params = {'T': T, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'Wf': Wf}


def forward_pass_variational(trainable_params, x1, x2, y1):
    """computes forward pass for variational lower bound approach"""

    out0_1 = tf.matmul(trainable_params['T'], x1)
    out0_2 = tf.matmul(trainable_params['T'], x2)

    yz1 = tf.concat([out0_1, y1], axis=0)
    yz2 = tf.concat([out0_2, y1], axis=0)  # use ys from first batch, z from second
    out1_1 = tf.nn.sigmoid(tf.matmul(trainable_params['W1'], yz1) + trainable_params['b1'])
    out1_2 = tf.nn.sigmoid(tf.matmul(trainable_params['W1'], yz2) + trainable_params['b1'])
    out2_1 = tf.nn.sigmoid(tf.matmul(trainable_params['W2'], out1_1) + trainable_params['b2'])
    out2_2 = tf.nn.sigmoid(tf.matmul(trainable_params['W2'], out1_2) + trainable_params['b2'])
    f_1 = tf.nn.sigmoid(tf.matmul(trainable_params['Wf'], out2_1))
    f_2 = tf.nn.sigmoid(tf.matmul(trainable_params['Wf'], out2_2))

    return f_1, f_2


f_1, f_2 = forward_pass_variational(trainable_params, X1, X2, Y1)
f_1v, f_2v = forward_pass_variational(trainable_params, X1v, X2v, Y1v)

loss = -tf.reduce_mean(f_1) + tf.log( tf.reduce_mean(tf.exp(f_2)) )
vloss = -tf.reduce_mean(f_1v) + tf.log( tf.reduce_mean(tf.exp(f_2v)) )

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    trainable = tf.trainable_variables()

    for i in range(num_steps):
        batch_x1, batch_y1 = get_minibatch(batch_size, x_train, y_train, False)
        batch_x2, _ = get_minibatch(batch_size, x_train, y_train, False)

        _, batch_loss, v_loss = \
            sess.run([optimizer, loss, vloss], feed_dict={X1: batch_x1, Y1: batch_y1, X2: batch_x2})
        batch_training_losses.append(batch_loss)
        validation_losses.append(v_loss)

####################################### POST TRAINING ###################################################
plt.plot(batch_training_losses)
plt.plot(validation_losses)
plt.show()
plt.clf()




