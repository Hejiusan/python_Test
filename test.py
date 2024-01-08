import matplotlib.pyplot as plt
import numpy as np

def cal_xbar(x):
    if x < 150:
        return 0.001205994950606339 * x ** 2 + 0.08040692427806707 * x + 34.473656243654254
    elif x < 180:
        return 0.3 * x + 26
    elif x < 200:
        return 0.6 * x - 23
    elif x < 270:
        return 0.004467535594029877 * x ** 2 - 2.423716859411953 * x + 398.86537740415025
    else:
        return 0.00012981895640459996 * x ** 2 - 0.2013928524633596 * x + 114.2570530071489

def cal_xaar(x):
    if x < 130:
        return 0.0007162963282137652 * x ** 2 + 0.029544017875577 * x + 34.55845339041658
    elif x < 200:
        return 0.0002701518810263071 * x ** 2 + 0.19521606856935872 * x + 27.64533781304466
    elif x < 220:
        return -0.02737574186688857 * x**2 + 12.021660324441497 * x + -1237.867051140388
    elif x < 290:
        return 0.0039762317520980256 * x ** 2 + -1.8200938632861114 * x + 280.7600812112186
    elif x < 300:
        return 0.037121678475383765 * x ** 2 + -22.54116104985299 * x + 3508.901908747426
    elif x < 400:
        return -0.0019543348142924094 * x ** 2 + 1.2128135231051356 * x -111.44797891476394
    else:
        return 5.149664249494104e-05 * x ** 2 + -0.12284111009268611 * x + 111.701029898968

def cal_rst(x):
    if x < 4.5:
        return 30
    elif x < 7.5:
        return 10 * x - 15
    elif x < 13.5:
        return 5 * x + 22.5
    elif x < 16:
        return 4 * x + 36
    else:
        return 30

def cal_xst(x):
    if x < 3:
        return 30
    elif x < 7.9:
        return 10 * x + 1
    elif x < 9.9:
        return 5 * x + 40.5
    elif x < 12.5:
        return 4 * x + 50
    else:
        return 31

def cal_xvla(x):
    if x < 9:
        return x + 31
    elif x < 16:
        return 2.86 * x + 14.286
    elif x < 38:
        return x + 44
    elif x < 50:
        return 0.417 * x + 64.17
    elif x < 130:
        return 0.125 * x + 78.75
    else:
        return 100

def cal_xsb(x,xst):

    if xst == 0:
        return 0
    if x < 4:
        return -5 * x + 100
    elif x < 8:
        return -2.5 * x + 90
    elif x < 11:
        return -3.33 * x + 96
    elif x < 18:
        return -4.14 * x + 105.57
    else:
        return 30


# # 生成 x 值的范围
# x = np.linspace(0, 800, 1000)
#
# # 计算对应的 y 值
# y = np.vectorize(cal_xaar)(x)
#
# # 绘制图像
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of the Function')
# plt.grid(True)
# plt.show()

x = 800
y = 1200
# xbar: 82.90084045573411 xaar: 42.81664917675744
# 62.858744816245775
xbar = cal_xbar(x)
xaar = cal_xaar(y)

if 120 < x < 150 or 180 < x < 200:
    xbar = xbar - 5


cardsc = (xbar + xaar)/2
print("xbar:", xbar,"xaar:",xaar)
print(cardsc)

# rst = 10.75
# xst = 8.08
# xvla = 37.12
# xsb = 7
#
# sleep = (cal_xsb(xsb,xst) + cal_xst(xst) + cal_rst(rst) + cal_xvla(xvla))/4
# print("xsb:", cal_xsb(xsb,xst), "xst:", cal_xst(xst), "rst:", cal_rst(rst), "xvla:", cal_xvla(xvla))
# print(sleep)