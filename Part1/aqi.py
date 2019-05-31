import numpy as np
import read_data
from show import show_cost
"""
演示梯度下降，梯度下降有几个重要因素：
1. 梯度（在单变量的函数中，梯度其实就是函数的微分，代表着函数在某个给定点的切线的斜率
         在多变量函数中，梯度是一个向量，向量有方向，梯度的方向就指出了函数在给定点的上升最快的方向）
2. 负梯度
3. 学习率
"""
def get_grad(theta, x, y):
    # 求梯度
    grad = np.dot(np.transpose(x), (np.dot (x, theta) - y))
    return grad # 因为梯度和方向是呈相反的，所以需要加一个负号

def gradient_descending(theta, x, y, v_x, v_y, learning_rate):
    # 通过梯度下降算法，对线性回归模型进行训练
    train_costs = [] # 记录训练过程中产生的cost
    validation_costs = [] # 记录验证集上产生的cost
    for _ in range(200):
        theta = theta - get_grad(theta, x, y) * learning_rate
        train_costs.append(get_cost(theta, x, y))
        validation_costs.append(get_cost(theta, v_x, v_y))
    show_cost(train_costs, validation_costs)
    # TODO: 将theta的值保存起来
    with open('model.txt', 'w') as f:
        for i in theta:
            for j in i:
                f.write(str(j) + "\n")
    return theta

def test_model(theta,test_x,test_y):
    # 使用R方差来测试模型的优劣
    r = 1 - get_cost(theta, test_x, test_y)/np.var(test_y) # np.var 计算方差
    print(r)

def get_cost(theta, x, y):
    # x: 是一个矩阵
    # y: 是一个矩阵
    # theta: 是一个矩阵
   return np.mean((np.dot(x, theta) -y) ** 2)  # np.mean是计算平均值

def get_aqi_value(input_data):
    # 根据用户提供的输入数据，完成aqi值的预测
    x = np.array(input_data)
    x = read_data.standard_data(x)
    # 从文件中读取theta
    with open('model.txt','r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6,1)
    return np.dot(x,theta)
"""
train_data, validation_data, test_data = read_data.read_aqi()
# x,y = read_data.read_aqi()
# print(x[5:])
theta = np.zeros((6,1))
learning_rate = 0.001
theta = gradient_descending(theta, train_data[0], train_data[1], validation_data[0], validation_data[1], learning_rate)

aqi_value = get_aqi_value([33,56,7,27,0.82,101])
print(aqi_value)
"""
"""
x = np.array([[1,2],[1,2]])
y = np.array([0,0])
theta = np.array([1,1])
print(get_cost(theta, x, y))
"""
"""
y = 20
x = 1.1
theta = 0
learning_rate = 0.1
gradient_descending(theta, x, y, learning_rate)
cost = get_cost(theta, x, y)
print(cost)
"""

