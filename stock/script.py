
import statsmodels.api as sm
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq


total_day = 34
train_day = 30
test_day = 4


stock_return = np.loadtxt("601666.csv", dtype=str, delimiter='"', skiprows=1, usecols=(13))
index_return = np.loadtxt("index.csv", dtype=str, delimiter='"', skiprows=1, usecols=(13))

print(stock_return)
print(index_return)

cut_stock = []
cut_index = []
for i in range(total_day):
    cut_stock.append(float(stock_return[i][:-1]) / 100)
    cut_index.append(float(index_return[i][:-1]) / 100)

stock_log = np.log10(np.array(cut_stock) + 1)
index_log = np.log10(np.array(cut_index) + 1)

y = stock_log[:30]
x = sm.add_constant(index_log[:30]) # 若模型中有截距，必须有这一步
model = sm.OLS(y, x).fit() # 构建最小二乘模型并拟合
print(model.summary()) # 输出回归结果
print(type(model.summary()))

# f test
ftest = model.f_test("const=0")
print(ftest)

