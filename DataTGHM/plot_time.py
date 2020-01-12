import numpy as np
import matplotlib.pyplot as plt

time = 3

time_tghm = [7.41, 6.98, 8.42, 8.58, 7.04, 8.81, 7.63, 8.36, 9.96, 7.26]
time_fs = [43.5, 36.0, 37.0, 34.8, 39.5, 36.4, 32.4, 36.4, 32.4, 44.2]
time_naive = [22.9, 24.2, 19.7, 25.7, 15.7, 18.4, 27.4, 20.4, 30.2, 19.2]
x = list(range(1, 11))

plt.figure()
line_tghm = plt.plot(x, time_tghm, 'g', label='TGHM', lw=2, marker='^', mew=3)
line_fs = plt.plot(x, time_fs, 'b', label='FS', lw=2, marker='*', mew=3)
line_naive = plt.plot(x, time_naive, 'r', label='Naive', lw=2, marker='s', mew=3)

plt.axis([0, 12, 0, 60])  # 坐标轴范围
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.legend()  # 显示图例
plt.xlabel("The number of experiments", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')  # 坐标轴标注
plt.ylabel("Time consumption (min)", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')
# plt.title('折线图', fontproperties='SimHei', fontsize=12)  # 标题
plt.savefig("simulation_time.svg", dpi=1800)
plt.show()

