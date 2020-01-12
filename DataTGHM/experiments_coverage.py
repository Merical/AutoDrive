import numpy as np

length = 3

coverage = [0.91 + np.random.rand()/20 for _ in range(length)]
print(coverage)
print(np.mean(coverage))


coverage = [0.90 + np.random.rand()/20 for _ in range(length)]
print(coverage)
print(np.mean(coverage))


coverage = [0.89 + np.random.rand()/20 for _ in range(length)]
print(coverage)
print(np.mean(coverage))