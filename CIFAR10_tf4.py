#CIFAR图像识别

#下载数据集
#导入官方CIFAR10模块
import cirfar10
import tensorflow as tf

#tf.app.flags.FLAGS是TF的一个内部全局变量存储器
FLAGS = tf.app.flags.FLAGS

#将预定义下载路径改为当前位置
FLAGS.data_dir = './cifar10_data'

#若不存在数据文件则下载，并且解压
cifar10.maybe_download_and_extract()

'''
数据读取DEMO
不打乱读取顺序的代码

import tensorflow as tf

filenames = ['1.jpg','2.jpg','3.jpg']

shuffle = False表示不打乱顺序，num_epochs=3表示整个队列获取3次
queue = tf.train.string_input_producer(filenames, shuffle=False,num_epochs=3)

读取文件名队列中的数据
reader = tf.WholeFileReader()
key,value = reader.read(queue)

with tf.Session() as sess:
    初始局部化变量，注意这个函数跟tf.global_variabless_initializer.run()是不一样的
    因为string_input_producer函数的num_epochs=3传入的是局部变量
    tf.local_variables_initializer().run()
    threads = tf.trian.start_queue_runners(sess=sess)
    i += 1
    data = sess.run(value)
    with open('shuffle_false/image_%d.jpg' % i,'wb') as fd:
    fd.write(data)

'''
def download():
    #tf.app.flags.FLAGS内部全局变量储存器
    FLAGS = tf.app.flags.FLAGS
    #为了方便，我们将这个路径改为当前位置
    FLAGS.data_dir = '.cirfar10_data'
    cirfar10.mabe_download_and_extract()

image_save_path = './cirfar10_image/'
if os.path.exists(image_save_path) == False:
    os.mkdir(image_save_path)
def check_cifar10_data_files(filesnames):
    for file in flienames:
        if os.path.exsits(file) ==False:
            print('not found cifar10 data.')
            return False
        return True

#获取图片前的预处理，检测CIFAR10数据是存在，若不存在直接退出
#如果存在，用string_input_producer 函数创建文件名队列
#通过get_record函数获取图片标签和图片数据，幷返回
def get_image(data_path):
    filenames = [0s.path.join(data_path,"data_batch_%d.bin" % i)for i in range(1,6)
    print (filenames)
    if check_cifar10_data_files(filesnames)
        exit()
    queue = tf.train.string_input_producer(queue)
    return get_record(queue)

def get_record(queue):
    print ('get_record')
    label_bytes = 1
    image_width = 32
    image_height = 32
    image_depth = 3
    image_bytes = image_width * image_height *image_depth
    record_bytes = label_bytes + image_bytes

    #根据样本大小读取数据
    reader = tf.FixedLengthRecordReader(record_bytes)
    key ,value = reader.read(queue)

    #将获取结果转变成一维数组
    record_bytes = tf.decode_raw(value,tf.unit8)
    
    #获取label,label数据在每个样本的第一个字节
    label_data = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes])

    #获取图片数据，label后到样本末尾的数据即图片数据
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes],
                         [3,32,32])
    #矩阵转置成RGBRGBRGB
    image_data = tf.transpose(depth_major,[1,2,0])
    image_data = tf.cast(image_data,tf.float32)
    return label_data, image_data
    
                         
if _name_ == '_main_':
    #查看CIFAR-10数据是否存在，如果不存在，则下载幷解压
    download()

    #将获取的图片保存
    image_save_path = './cifar10_image/'
    if os.path.exists(iamge_save_path) == False:
        os.mkdir(image_save_path)
    #获取图片数据
    key, value = get_image('./cifar10_data/cifar-10-batch-bin/')

    with tf.Session() as sess:
        sess.run(tf.global_variable_intializer())
        coord = tf.train.Coordinator()
        #这里才真的启动队列
        threads = tf.train.start_queue_runners(sess=sess,coord = coord)

        for i in range (50):
            #这里data和label不能分开run,否则图片和标签就不匹配了
            label, data = sess.run([key,value])
            print(label)
            scipy.misc.toimage(data).save(image_save_path +  '/%d_%d.jpg' % (label,i))
        coord.request_stop()
        coord.join()
        
