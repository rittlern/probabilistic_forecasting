

import numpy as np

num_exp = 1000
A = np.diag([10,2,2,2])
A = A + np.reshape(np.random.normal(1, 1, 16), (4,4))

x = np.zeros([4])
u = np.random.multivariate_normal([0,0,0,0], cov=np.identity(4))  # random unit direction
u = u * (1/np.linalg.norm(u))
mu = np.matmul(A, x)
Sig = np.eye(4) + np.matmul(A, np.transpose(A))


def integrand(t, x, A):
    Sig = np.eye(4) + np.matmul(A, np.transpose(A))
    v = np.matmul(np.matmul(np.transpose(A), np.linalg.inv(Sig)), (np.matmul(A, x) - t) )
    return np.abs( np.matmul(np.transpose(u), v ) )

### generate normal noise
noise = np.random.multivariate_normal(mean=mu, cov=Sig, size=num_exp)
evals = np.empty(num_exp)

for i in range(num_exp):

    evals[i] =integrand(noise[i, :], x, A)

print(np.mean(evals))
