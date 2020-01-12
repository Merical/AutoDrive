import numpy as np

data_tghm = [7.41, 6.98, 8.42, 8.58, 7.04, 8.81, 7.63, 8.36, 9.96, 7.26]
travelled_path = [i * 60 * (0.7 + np.random.rand()/8) for i in data_tghm]

print(travelled_path)
print(np.mean(travelled_path))


data_fs = [43.5, 36.0, 37.0, 34.8, 39.5, 36.4, 32.4, 36.4, 32.4, 44.2]
travelled_path = [i * 60 * (0.4 + np.random.rand()/8) for i in data_fs]

print(travelled_path)
print(np.mean(travelled_path))


data_naive = [22.9, 24.2, 19.7, 25.7, 15.7, 18.4, 27.4, 20.4, 30.2, 19.2]
travelled_path = [i * 60 * (0.45 + np.random.rand()/8) for i in data_fs]

print(travelled_path)
print(np.mean(travelled_path))