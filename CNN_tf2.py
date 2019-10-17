#CNN模型实现MNIST手写数字识别--tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist as input_data

mnist = input_data.read_data_sets('mnist_data', one_hot = True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

#将图片从784维向量还原为28×28的矩阵图片
# 原因参考卷积神经网络模型图，最后一个参数代表深度
# 因为MNIST是黑白图片，所以深度为1
# 第一个参数为-1,表示一维的长度不限定，这样就可以灵活设置每个Batch的训练个数了
x_image = tf.reshape(x,[-1,28,28,1])

#设置过滤器和偏置初始化函数
#初始化过滤器
def weight_variable(shape):
    return tf.Variable(tf.truncated_nomal(shape,stddev=0.1))
    #tf.truncated_nomal(shape,mean,stddev)
    #shape:表示生成张量的维度；mean:均值; stddev :标准差
    #此函数产生正态分布，截断的产生正态分布

#初始化偏置，所有值为0.1
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape = shape))
    #创建一个常量

#设置卷积和池化运算函数
#卷积运算，strides表示每一维度滑动的步长，一般strides[0]=strides[3]=1
#第四个参数可选“same"/"VALID","same"表示边距使用全0填充
def conv2d(x,w):
    return tf.nn.conv2d(x,w,srtides[1,1,1,1],padding="Same")
    '''
    tf.nn.conv2d (input, filter, strides, padding,
    use_cudnn_on_gpu=None, data_format=None, name=None)
    input:输入要做卷积的图片，要求为一个张量，shape:[batch,in_height,in_width,in_channel]
    filter:卷积核，要求也是一个张量，shape:[filter_height,filter_width,in_channel,out_channel]
    strides:卷积时在图像每一维的步长，一维向量 [1,strides,strides,1]
    padding:string类型，值为SAME 和VALID，表示卷积的形式，是否考虑边界
            ”SAME“是考虑边界，不足的时候用0去填充周围
    use_cudnn_on_gpu:bool类形，是否使用cudnn加速，默认为真
    '''
#池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    '''
    tf.nn.max_pool(value, ksize, strides, padding, name=None)
    value:需要池化的输入，一般池化层接在卷积层后面，输入通常为feature map
            [batch,height,width,channels]
    ksize:池化窗口的大小，取一个四维向量，一般是[1,height,width,1]
            因为我们不想在batch和channel上做优化，所以这两个维度设为1
    strides:窗口在每一个维度上滑动的步长，一般是[1,stride,stride,1]
    padding:和卷积类似，可以取VALID或者SAME
    '''
    
#第一层卷积 
#将过滤器设置成5×5×1的矩阵
#5*5表示过滤器大小，1表示深度，MNIST黑白图片只有一层
#32表示我们要创建32个大小5×5×1的过滤器，经过卷积后算出32个特征图（每个过滤器得到一个特征图）
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#使用conv2d函数进行第一次卷积计算,然后再用RELU作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
h_pool1 = max_pool2x2(h_conv1)

#第二层卷积
#因为经过第一层卷积运算后，输出的深度为32,所以过滤器深度和下一层输出深度也做出改变
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool,w_conv2) + b_conv2)
h_pool2 = max_pool2x2(h_conv2)

#全连接层
#经过两层卷积后，图片的大小为7×7（第一层池化后输出为（28/2）×（28/2），
#第二层池化后输出为（14/2）×（14×2））深度为64
#我们在这里加入一个有1024个神经元的全连接层    权重W尺寸为[7*7*64,1024]
w_fc1 = weight_varable([7*7*64,1024])
#偏置的个数和权重个数一致
b_fc1 = bias_variable([1024])
#这里将第二层池化后的张量（7，7,64） 变成向量，跟上一节softmax输入一样
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

#dropout
#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.palceholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
'''
tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None)
为了防止或减轻过拟合而使用的函数，一般用在全连接层
DROPOUT就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值
以一定的概率让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算
但他的权重得以保留下来（只是暂时不更新），因为下次样本输入又得工作了
x:输入tensor
keep_prob:float类型，每个元素被保留下来的概率，设置神经元被选中的概率
            初始化时是一个占位符，运行时设置具体的值
noise_shape:一个1维int32张量，代表随机产生”保留或丢弃的shape"
seed:int 随机数种子
name:指定该操作的名字
'''

#输出层
#全连接层输入大小为1024,而结果大小为10
#所以权重W尺寸为[1024,10]
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.sotfmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#损失函数和损失优化
cross_entropy = tf.reduce(-tf.reduce.sum(y_ * tf.log(y_conv)))
trian_step = tf.trian.AdamOptimizer(1e-4).minimize(cross_entropy)


#测试准确率，跟SOFTMAX回归模型一样
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#保存训练结果
savePath = './mnist_conv/'
saveFile = savePath + 'mnist_conv.ckpt'
if os.path.exists(savePath) == False:
    os.mkdir(savePath)

saver = tf.train.Saver()
....
saver.save(sess, saveFile)

#开始训练
with tf.Session() as sess:
    #初始化所有变量
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        #每次获取50张图片数据和对应的标签
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            trian_accuracy = sess.run(accuracy, feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d,training accuracy %g",%(i,train_accuracy))
        #这里是真的训练，将数据传入
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    #训练结束后，使用测试集测试最后的准确率
    print("test accuracy %g" % sess.run(accuracy,feed_dict={x:mnist.test.images,
                                                            y_:mnist.test.labels,keep_prob:1.0})
        
    #最后保存会话
    saver.save(sess,saveFile)
                                   
#GPU显存不够时，将整个测试数据拆分成多个小的batch进行测试
          mean_value = 0.0
          for i in range(mnist.test.labels.shape[0]):
                batch = mnist.test.next_batch(50)
                train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
                mean_value += train_accuracy
          print("test accuracy %g" % (mean_value/mnist.test.labels.shape[0])
          
