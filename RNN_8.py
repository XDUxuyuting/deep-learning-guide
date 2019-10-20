#循环神经网络模仿二进制减法
#用python写的，没用tf框架

import copy
import numpy as np

#定义sigmoid函数
def sigmoid(x):
    output = 1/(1+ np.exp(-x))
    return output

#定义sigmoid函数的导数，用于计算梯度下降
#output是sigmoid函数的输出
def sigmoid_output_to_derivative(output):
    return output * (1-output)

#定义十进制数转二进制映射
#二进制的位数，这里只计算8位
binary_dim = 8
#8位二进制的最大数，2的8次方
largest_number = pow(2,binary_dim)

#用于整数到二进制表示的映射
#比如十进制数2的二进制表示，可以写为int2binary[2]
int2binary = {}

#将十进制数0-255转成二进制表示
#再将其存到int2binary中
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

#设置学习率、输入层维度、隐藏层维度、输出层维度
#学习率
learning_rate = 0.9
#循环体的输入的维度，比如计算11 - 6,其二进制形式分别对应
# 00001011
# 00000110
#这里的输入分别是这两个数对应下标的bit
input_dim = 2
#循环体内隐藏的维度
hidden_dim = 16
#输出的维度
output_dim = 1

#定义权重幷初始化，定义用于反向传播权重梯度值的变量
#定义神经网络的权重的形状幷初始化
#w_input_hidden 输入层到隐藏层的权重
#w_hidden_output 是隐藏层到输出层的权重
#w_hidden 隐藏层的权重
#输入层(1,2)，输出层（1），隐藏层（1,16）
#w_input_hidden (2,16) w_hidden_output (16,1) w_hidden (16,16)
#训练过程就是更新这三个权重的过程
w_input_hidden = (2 * np.random.random((input_dim,hidden_dim)) - 1) * 0.05
w_hidden_output = (2 * np.random.random((hidden_dim,output_dim)) -1) * 0.05
w_hidden = (2 * np.random.random((hidden_dim, hidden_dim)) - 1) * 0.05

#用于存放反响传播的权重梯度值
w_input_hidden_update = np.zeros_like(w_input_hidden)
w_hidden_output_update = np.zeros_like(w_hidden_output)
w_hidden_update = np.zeros_like(w_hidden)

#开始训练
for j in range(10000):
    #生成一个被减数，[0,256)
    a_int = np.random.randint(largest_number)
    #生成减数，[0,128)
    b_int = np.random.randint(largest_number/2)
    
    #如果被减数比减数小则互换
    #我们暂时不考虑负数，所以确保被减数比减数大
    if a_int < b_int:
        tmp = a_int
        b_int = a_int
        a_int = tmp

    #转化为二进制
    a = int2binary[a_int]
    b = int2binary[b_int]

    #c保存a-b的二进制答案
    c_int = a_int - b_int
    c = int2binary[c_int ]

    #存储神经网络的预测值的二进制形式
    d = np.zeros_like(c)

    #定义用来存储循环体输出层的误差倒数和隐藏层的值的list
    #存储每个循环体输出层的误差倒数
    layer_output_deltas = list()

    #存储每个循环体隐藏层的值
    layer_hidden_values = list()

    #初始化总误差
    over_all_error = 0

    #一开始没有隐藏层，所以初始化一下原始值为0.1
    layer_hidden_values.append(np.ones(hidden_dim) * 0.1)

    #前向传播，循环遍历每一个二进制位
    #隐藏层：layer_hidden = sigmoid(np.dot(X,w_input_hidden) +np.dot(layer_hidden_values[-1], w_hidden))
    #输出层 layer_output = sigmoid(np.dot(layer_hidden, w_hidden_output))
    for position in range(binary_dim):

        #从低位开始，每次取被减数和减数的一个bit作为循环体的输入
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        #这里取答案c想应的bit位，作为与预测值的对比， 以取得预测误差
        y = np.array([[c[binary_dim - position - 1]]]).T

        #计算隐藏层，新隐藏层 = 输入层+旧隐藏层
        #这里 X是循环体输入，layer_hidden_values[-1]是上一个循环体的隐藏层的值
        #从这里可以看出，循环体的输出不只跟本次的输入有关，还跟上一个循环体也有关
        layer_hidden = sigmoid(np.dot(X, w_input_hidden) + np.dot(layer_hidden_values[-1],w_hidden))

        #输出层，本次循环的预算值
        layer_output = sigmoid(np.dot(layer_hidden, w_hidden_output))
        
        #预测值与实际值的误差
        layer_output_error = y - layer_output

        #把每一个循环体的误差导数都保存下来
        layer_output_deltas.append((layer_output_error) * sigmoid_output_to_derivative(layer_output))
        
        #计算总误差
        over_all_error += np.abs(layer_output_error[0])

        #保存预测bit位，这里对预测值进行四舍五入，保存的值要么是0,要么是1
        d[binary_dim - position -1] = np.round(layer_output[0][0])

        #保存本次循环体的隐藏层，供下个循环体使用
        layer_hidden_values.append(copy.deepcopy(layer_hidden))


        #反向传播
        future_layer_hidden_delta = np.zeros(hidden_dim)

        #反向传播，从最后一个循环体到第一个循环体
        for position in range(binary_dim):
            #获取循环体的输入
            X = np.array([[a[position], b[position]]])

            #当前循环体的隐藏层、
            layer_hidden = layer_hidden_values[-position -1]

            #上一个循环体的隐藏层
            prev_layer_hidden = layer_hidden_values[-position -2]

            #获取当前循环体的输出误差导数
            layer_output_delta = layer_output_deltas[-position -1]

            #计算当前隐藏层的误差
            #通过后一个循环体（因为是反向传播）的隐藏层数和当前循环体的输出层误差，计算当前循环体的隐藏层误差
            layer_hidden_delta = (future_layer_hidden_delta.dot(w_hidden.T)+
                                  layer_output_delta.dot(w_hidden_output.T)) * sigmoid_output_to_derivative(layer_hidden)
            
            #等到完成了所有的反向传播误差计算，才会更新权重矩阵，先暂时把更新矩阵存起来
            w_input_hidden_update += X.T.dot(layer_hidden_delta)
            w_hidden_output_update += np.atleast_2d(layer_hidden).T.dot(layer_output_delta)
            w_hidden_update += np.atleast_2d(prev_layer_hidden).T.dot(layer_hidden_delta)

            future_layer_hidden_delta = layer_hidden_delta

        #完成所有反向传播后，更新权重矩阵，并把矩阵变量清零
        w_input_hidden += w_input_hidden_update * learning_rate
        w_hidden_output += w_hidden_output_update * learning_rate
        w_hidden += w_hidden_update * learning_rate
        w_input_hidden_update *= 0
        w_hidden_update *= 0
        w_hidden_update *= 0

        #每800次打印一次训练结果
        if (j % 800 == 0):
            print ("all error:" + str(over_all_error))
            print ("Pred:" + str(d))
            print ("True:" + str(c))

            #将二进制取转化成十进制
            for index, x in enumerate(reversed(d)):
                out += x * pow(2, index)

            print (str(a_int) + " - " + str(b_int) + " = " + str(out))
            print ("-------------")
    
        
