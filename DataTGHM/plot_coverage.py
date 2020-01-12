import numpy as np
import matplotlib.pyplot as plt

length = 3

coverage_tghm = [97.9, 98.1, 98.7, 94.1, 96.4, 96.1, 96.6, 95.8, 98.2, 94.3]
coverage_fs = [95.9, 94.2, 92.2, 95.8, 93.8, 92.9, 95.4, 94.5, 96.2, 92.6]
coverage_naive = [92.2, 92.6, 91.9, 94.4, 92.3, 93.4, 93.9, 92.5, 94.7, 93.3]
x = list(range(1, 11))

plt.figure()
line_tghm = plt.plot(x, coverage_tghm, 'g', label='TGHM', lw=2, marker='^', mew=3)
line_fs = plt.plot(x, coverage_fs, 'b', label='FS', lw=2, marker='*', mew=3)
line_naive = plt.plot(x, coverage_naive, 'r', label='Naive', lw=2, marker='s', mew=3)

plt.axis([0, 12, 90, 100])  # 坐标轴范围
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.legend()  # 显示图例
plt.xlabel("The number of experiments", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')  # 坐标轴标注
plt.ylabel("Coverage Rate (%)", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')
# plt.title('折线图', fontproperties='SimHei', fontsize=12)  # 标题
plt.savefig("simulation_coverage.svg", dpi=1800)
plt.show()

