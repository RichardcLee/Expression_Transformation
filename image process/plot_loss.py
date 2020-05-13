from matplotlib import pyplot as plt
import re


plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

lines = []
with open(r"C:\Users\81955\Desktop\logs\logs.txt", "r+") as f:
    lines = f.readlines()


# dis_fake WGAN-GP对抗损失第二项，值越大越好（正值）
# dis_real WGAN-GP对抗损失第一项，值越小越好（负值）
# dis_real_aus 条件表情损失第二项
# gen_rec 循环一致性损失

loss = {
    "dis_fake": [],
    "dis_real": [],
    "dis_real_aus": [],
    "gen_rec": [],
    # 'dis': [],
    # 'gen': [],
    "total": []
}

for line in lines:
    a, b, c, d = float(re.findall("dis_fake:(.*?)\|", line)[0].strip()), float(re.findall("dis_real:(.*?)\|", line)[0].strip()), float(re.findall("dis_real_aus:(.*?)\|", line)[0].strip()), float(re.findall("gen_rec:(.*?)\|", line)[0].strip())
    loss["dis_fake"].append(a)
    loss["dis_real"].append(b)
    loss["dis_real_aus"].append(c)
    loss["gen_rec"].append(d)
    loss["total"].append(10*d + 1*(a+b) + 160*c)

print(loss)

plt.figure(dpi=120)
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.45) # 调整子图间距
xy = ["221","222", "223", "224"]
widths = [0.09, 0.09, 0.10, 0.15]
labels = ['adversarial loss 2', 'adversarial loss 1', 'condition loss', 'cycle consistency loss', 'total loss']
ticks_y = [[0, 1, 2, 3, 4], [-5, -4, -3, -2, -1], [0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.022, 0.024, 0.026], [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]]
ticks_x = ['0', '2k', '4k', '6k', '8k', '1w']
scale_x = [0, 1000, 2000, 3000, 4000, 5000]
idx = 0
step = [i for i in range(len(loss["dis_fake"]))]
fontsize = 10

for name, val in loss.items():
    if idx == 4:
        continue
    plt.subplot(xy[idx])
    plt.title(labels[idx], fontsize=fontsize+2)
    plt.plot(step[::2], val[::2], linewidth=widths[idx], color='k')  # label=labels[idx]
    # plt.legend(loc='best')
    plt.xlabel("step", fontsize=fontsize-1)
    plt.ylabel("loss value", fontsize=fontsize-1)
    # 设置刻度字体大小
    plt.xticks(scale_x, ticks_x, fontsize=fontsize-1)
    plt.yticks(ticks_y[idx], fontsize=fontsize-1)
    idx += 1

fontsize = 20
plt.figure(dpi=80)
plt.plot(step[::2], loss['total'][::2], linewidth=0.2, color='k')
plt.xlabel("step", fontsize=fontsize-6)
plt.ylabel("loss value", fontsize=fontsize-6)
# 设置刻度字体大小
plt.xticks(scale_x, ticks_x, fontsize=fontsize-6)
plt.yticks(fontsize=fontsize-1)

# plt.show()
plt.savefig("1.jpg")

