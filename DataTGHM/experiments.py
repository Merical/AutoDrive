import numpy as np

data_tghm = [2.51, 2.46, 2.78]
travelled_path = [i * 60 * (0.4 + np.random.rand()/8) for i in data_tghm]

print(travelled_path)
print(np.mean(travelled_path))


data_fs = [4.52, 4.79, 5.06]
travelled_path = [i * 60 * (0.3 + np.random.rand()/8) for i in data_fs]

print(travelled_path)
print(np.mean(travelled_path))


data_naive = [4.47, 5.84, 5.23]
travelled_path = [i * 60 * (0.35 + np.random.rand()/8) for i in data_fs]

print(travelled_path)
print(np.mean(travelled_path))