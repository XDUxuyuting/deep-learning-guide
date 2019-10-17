#softmax回归模型训练MNIST

#将MNIST数据集保存为图片
from tensorflow.examples.tutorials.mnist import input_data
import os
import scipy.misc as sm
import numpy as np

mnist = input_data.read_data_sets("mnist_data", one_hot =True)
save_dir = 'mnist_data/image/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

for i in range(50):
    image_array = mnist.train.images[i, :]
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    image_array = image_array.reshape[28,28]
    filename = save_dir +'image_train_%d_%d.jpg' %(i,label)
    sm.toimage(image_array).save(filename)

#tensorflow代码实现

    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data', one_hot = True)

#创建占位符，用于临时存放MNIST图片的数据
# [None,784]中的None表示不限长度，而784则是一张图片的大小
x = tf.placeholder(tf.float32,[None,784])

#w存放模型参数，也就是权重，一张图片有784个像素作为输入数据，而输出为`10
#0～9有10个结果
#b存放偏置项
w = tf.Variable(tf.zeros[784,10])
b = tf.Variable(tf.zeros(10))

#y表示Softmax回归模型的输出
y = tf.nn.softmax(tf.matmul(x,w)+b)
#y_是实际图像的标签，即对应于每张输入图片实际的值
y_= tf.palceholder(tf.float32, [None,10])

#定义损失函数，这里用交叉熵来做损失函数，y存的是训练结果，而y_存实际结果
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))

#优化函数，这里使用梯度下降法进行优化，0.01表示梯度下降优化器的学习率
tiran_step = tf.trian.GradientDescentOptimizer(0.01).minmize(cross_entropy)

#将训练结果保存，如果不保存，这次训练后的结果也随程序运行结束而释放
saver = tf.train.Saver()

#上面做的只是定义算法，并没有真的运行，tensorflow的运行都是在会话中进行
with tf.Session() as sess:
    #初始化所有变量
    tf.global_variables_initializer().run()

    #开始训练
    for _ in range (1000):
        #每次读取100张图片数据和对应的标签用于训练
        batch_xs,batch_ys = mnist.train.next_batch(100)
        #将读到的数据进行训练
        sess.run(train_step, feed_dict = {x:batch_xs,y:batch_ys})

    print(sess.run(w))
    print(sess.run(b))

    #检测训练结果，tf.argmax取出数组中最大值的下标，tf.equal再对比下标是否一样
    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
    #correct_prediction得到真假数组，在经过tf.cast转化成0-1数组
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels}))

    #最后保存会话
    saver.save(sess,'./saver/mnist.ckpt')


    #使用训练结果直接进行预测
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    mnist = input_data.read_data_sets('mnist_data',one_hot =True)

    x = tf.placeholder(tf.float32,[None,784])
    w = tf.Variable(tf.zeros[784,10])
    b = tf.Variable(tf.zeros(10))

    y = tf.nn.softmax(tf.matmul(x,w)+b)
    y_ = tf.placeholder(tf.float32,[None,10])

    saver = tf.train.Saver();

    with tf.Session() as sess:
        saver.restore(sess, './saver/mnist.ckpt')
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print (sess.run(accuracy, feed_dict = {x:mnist.text.images,y:mnist.test.labels}))
                                  
