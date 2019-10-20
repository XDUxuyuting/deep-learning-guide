#用tensorflow训练CIFAR10模型


#解压和下载CIFAR10数据集
def main(argv=None):
    cirfar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecurively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

#train函数
def train():
    """train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        '''
        tf.Graph() 表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，通俗来讲就是：
        在代码中添加的操作（画中的结点）和数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，
        你可以利用很多线程生成很多张图，但是默认图就只有一张。
        tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图，
        如果只有一个主线程不写也没有关系，tensorflow 里面已经存好了一张默认图，可以使用
        tf.get_default_graph() 来调用（显示这张默认纸），当你有多个线程就可以创造多个tf.Graph()，
        就是你可以有一个画图本，有很多张图纸，这时候就会有一个默认图的概念了。
        ''''
        
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
            #获取数据集的图片数据和对应的标签
            logits = cifar10.inference(images)
            #卷积模型的重点

...
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    iamges,labels = cifar10_input.distorted_inputs(data_dir = data_dir,
                                                   batch_size = FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images,tf.float16)
        labels = tf.cast(labels,tf.float16)

    return images,labels
    #将CIFAR10数据集的路径和batch大小传到cifar10_input.dostorted_inputs函数，再返回图片和标签的数据

#这里应该是有一个cifar10_input类
def distorted_inputs(data_dir,batch_size):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i)
                 for i in xrange(1,6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' +f)
    #创建一个可读文件名队列
    filename_queue = tf.train.string_input_producer(filenames)
    #因为数据集图片和标签的数据实际上放在data_batch_1.bin~data_batch_5.bin里，所以现将其放到
    #数组filnames里，然后传给tf.train.string_input_producer函数
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)
    #read_cifar10 返回一个类，而标签数据存在read_input.label,图片数据存在read_input.uint8image
    #再转换成浮点形

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image,[heoight,width,3])
    distorted_image = tf.random_flip_left_right(distorted_image)
    distorted_image = tf.random_brightness(distorted_image,max_delta = 63)
    distorted_imafe = tf.random_contrast(distorted_image,lower = 0.2,upper = 1.8)
    float_image = tf.per_image_standardization(distorted_image)

    #设置张量类型
    float_image.set_shape([height,eidth,3])
    read_input.label.set_shape([1])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLKES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examplesin_queue)
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples
                                           ,batch_size,shuffle = True)
    #以上为对图片数据进行增强操作

def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images,label_batch = tf.train.shuffle_batch(
            [image,label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples +3*batch_size,
            min_after_daqueue = min_queue_examples)
    else:
        images,label_batch = tf.train.batch(
            [image,label],
            batch_size = batch_size,
            num_threads=num_preprocess_threads,
            capacity = min_queue_examples + 3*batch_size)
    #输出训练图片
    tf.summary.image('images',images)

    return images,tf.reshape(label_batch,[batch_size])
    '''
    主要看tf.train.shuffle_batch函数，该函数主要输出一个打乱顺序的样本batch
    [image,label]表示样本和样本标签
    batch_size是样本batch长度
    capacity队列容量
    num_threads表示开启多少个线程
    min_after_dequeue表示出队后，队列中最少有这些个数据
    经过这些运算后，得到图片数据为一个四维张量[batch_size,height,width,3]
    标签为一维向量[batch_size]
    '''
def inference(images):
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5,5,3,64],
                                             stddev=5e-2,
                                             wd=None)#为变量添加weight_decay
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases= _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activition,name = scope.name)
        _activation_summary(conv1)#为激发函数的输出添加summary
        

    #pool1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='SAME',name='pool1')
    #norm1局部响应归一化
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,
                      name='norm1')
    #conv2
    with tf.bariable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5,5,64,64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_intializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = scope.name)
        _activation_summary(conv2)

    #norm2
    norm2 = tf.nn.lrn(conv2, 4, bias= 1.0, alpha = 0.001/9.0, bata=0.75,
                      name='norm2')
    #pool2
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],
                           strides=[1,2,2,1],padding='SAME',name='pool2')
    #local3
    with tf.variable_scope('local3') as scope:
        #Move everything into depth so we can perform a single matrix multiply
        reshape = tf.reshape(pool2,[images.get_shape().as_list()[0],-1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',shape=[dim,384],
                                              stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer.name)
        _activation_summary(local3)

    #local4
    with tf.variable_scope() as scope:
        weights = _variable_with_weight_decay('weights',shape=[384,192],
                                              stddev=0.4,wd=0.0004)
        biases = _variable_on_cpu('biases', [192], tf.constant_intializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +biases, name=scope.name)
        _activation_summary(local4)

        #线性层（wx+b)
     with tf.variable_scope('softmax_linear') as scope:
         weights = _variable_with_weight_dacay('weights', [192,NUM_CLASSES],
                                               stddev=1/192.0, wd=None)
         biases = tf._variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
         _activation_summary(softmax_linear)

    return softmax_linear
    #第一层卷积、池化，第二层卷积、池化，在经过三层全连接层

#计算损失函数
loss = cifar10.loss(logits, labels)
def loss(logits, labels):
    label = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, nmae='cross_entropy')
    #计算logits和labels的softmax交叉熵，在tf.reduce_mean求均值，再tf.add_n求和
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)#L2 loss

    return tf.add_n(tf.get_collection('losses')

trian_op = cifar10.train(loss, global_step)
def train(total_loss, global_step):
    #影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_PER_EPOCH_FOR_DECAY)

    #基于训练步数指数衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DACAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate',lr)

    #计算损失的滑动平均
    loss_averages_op = _add_loss_summaries(total_loss)

    #梯度下降法
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #添加训练变量的矩形图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    #添加梯度的矩形图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    #跟踪所有变量的滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DACAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variable())

    return variables_averages_op
        
class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime 打印时间、步数、损失值"""

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def  before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)

    def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            exampes_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sex; %.3f'
                          'sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value,
                                 examples_per_sec, sec_per_batch))
                    
with tf.train.MonitoredTrainingSession(
    checkpoint_dir=FLAGS.train_dir,
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
           tf.train.NanTensorHook(loss),
           _LoggerHook()],
    config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)) as mon_sess:
    while not mon_sess.should_stop():
        mon_sess.run(train_op)
    #tf.train.MonitoredTrainingSession 自动保存和载入模型的文件，默认每10分钟保存一次、
    #checkpoint_dir 传入保存的路径
    #tf.train.StopAtStepHook函数指定训练多少步后就停止
    #tf.train.NanTensorHook 用于监控loss,如果loss是Nan，则停止训练
    #_LoggerHook 则用于打印时间、步数、损失值等
    
    
