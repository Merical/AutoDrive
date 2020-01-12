import numpy as np
import matplotlib.pyplot as plt

length = 3

length_tghm = [325, 315, 401, 403, 328, 431, 346, 377, 440, 349]
length_fs = [660, 471, 525, 589, 527, 484, 439, 546, 477, 635]
length_naive = [617, 638, 536, 669, 490, 526, 614, 550, 625, 517]
x = list(range(1, 11))

plt.figure()
line_tghm = plt.plot(x, length_tghm, 'g', label='TGHM', lw=2, marker='^', mew=3)
line_fs = plt.plot(x, length_fs, 'b', label='FS', lw=2, marker='*', mew=3)
line_naive = plt.plot(x, length_naive, 'r', label='Naive', lw=2, marker='s', mew=3)

plt.axis([0, 12, 200, 800])  # 坐标轴范围
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.legend()  # 显示图例
plt.xlabel("The number of experiments", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')  # 坐标轴标注
plt.ylabel("Travelled Path Length (m)", fontproperties='Times New Roman', fontsize=18, fontweight='heavy')
# plt.title('折线图', fontproperties='SimHei', fontsize=12)  # 标题
plt.savefig("simulation_length.svg", dpi=1800)
plt.show()

