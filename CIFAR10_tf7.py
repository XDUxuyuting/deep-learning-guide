import tensorflow as tf
import os
import cifar10
import numpy as np

NUM_EXAMPLES_PER_EPOCH_FOR_TRIAN = 500
batch_size = 128
height = 24
width = 24

#查看CIFAR10数据是否存在，如果不存在则下载幷解压
def download():
    #tf.app.flags.FLAGS 内部全局变量存储器
    FLAGS = tf.app.flags.FLAGS
    #将路径改为当前位置
    FLAGS.data_dir = './cifar10_data'
    #若不存在，则下载并且解压
    cifar10.mabe_download_and_extract()

#检查数据是否存在
def check_cifar10_data_files(filenames):
    for file in filenames:
        if os.path.exists(file) == False:
            print ('not found cifar10 data')
            return False
        return True
#读取每个样本数据，样本由一个标签+一张图片组成
def get_record(queue):
    print('get_record')
    #定义label大小、图片宽度、高度、深度、图片大小、样本大小
    label_bytes = 1
    image_width = 32
    image_height = 32
    image_depth = 3
    image_bytes = image_width * image_height * image_depth
    record_bytes = label_bytes +image_bytes

    #根据样本大小读取数据
    reader = tf.FixedLengthRecordReader(record_bytes)
    key, value = reader.read(queue)

    #将获取的数据转变成一维数组
    #例如
    #source = 'abcde'
    #record_bytes = tf.decode_raw(source, tf.uint8)
    #运行结果为 97,98,99,100,101
    record_bytes = tf.decode_raw(value, tf.uint8)

    #获取label,label在每个样本的第一个字节
    label_data = tf.cast(tf.strided_slice(record_bytes, [0],
                                          [label_bytes], tf.int32))

    #获取图片数距，label后到样本末尾的数据即图片数据
    #再用tf.reshape函数将图片数据变成一个三位矩阵
    depth_major = tf.reshape(
        tf.stride_slice(record_bytes, label_bytes], [label_bytes+image_bytes],
        [3, 32, 32])
    #矩阵转置，上面得到的矩阵形式是[depth, height, width]即红绿蓝分别属于个维度
    image_data = tf.transpose(depth_major, [1,2,0])
    return label_data, image_data

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3*batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3*batch_size)

    #视觉显示图片
    tf.summary.image('images',images)
    return images, tf.reshape(label_batch, [batch_size])

#获取图片预处理，检测CIFAR10是否存在，若不存在直接退出
#如果存在，string_input_producer创建文件名队列
#通过get_record获取图片标签和图片数据返回
def get_image(data_path):
    filenames = [os.path.join(data_path,"data_batch_%d.bin" % i) for i in range
                 (1,6)]
    print (filenames)
    if check_cifar10_data_files(filenames) == False:
        exit(_)

    #创建文件名队列,获取标签图像数据，转换类型
    queue = tf.train.string_input_producer(filenames)
    label, image = get_record(queue)
    reshaped_image = tf.cast(image, tf.float32)
    
    #数据增强操作
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3] )
    distorted_image = tf.random_filp_left_right(distorted_image)
    dietorted_image = tf.random_brightness(distorted_image)
    distorted_image = tf.random_contrast(ditorted_image,
                                         lower=0.2, upper=1.8)

    #对图片标准化处理
    float_image = tf.image.per_image_standardization(distorted_image)

    #设置张量类型
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)

#初始化过滤器
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

#初始化偏置
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

#卷积运算，一般strides[0]=strides[3]=1
def  conv2d(s,W):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

if __name__ == '__mian__':
    download()

    images, labels = get_image('./cifar10_data/cifar-10-batches-bin/')

    x = tf.placeholder(tf.float32, [None, height, width, 3])
    y_ = tf.placeholder(tf.float32, [None, 10])

    #第一层卷积
    #32表示卷积在经过每个5×5大小的过滤器可以算出32个特征，输出深度为32
    w_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #第二层卷积
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #全连接层
    w_fc1 = weight_variable([6 * 6 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6* 6*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #输出层
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    #损失函数和损失优化
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
    train_step = tf.train.AdamOPtimizer(1e-4).minimize(cross_entropy)

    #测试准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrext_prediction, tf.flaot32))

    #保存训练结果
    savePath = './mycifar_conv/'
    savefile = savePath + 'mycifar_conv.ckpt'
    if os.path.exists(savePath) == False:
        os.mkdir(savePath)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #初始化变量
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners()
        for i range(15000):
            label_batch, image_batch = sess.run([labels, images])
            label_batch_onehot = np.eye(10, dtype=float)[label_batch]
            #生成对角矩阵

            sess.run(train_step, feed_dict={x:iamge_batch,
                                            y_:label_batch_onehot,
                                            keep_prob:1.0})
            if i % 10 == 0:
                result = sess.run(accuracy, feed_dict={x:image_batch,
                                                       y_:label_batch_onehot,
                                                       keep_prob:1.0})
                print('-----')
                print (result)
        saver.save(sess, saveFile)
            

