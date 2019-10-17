#两层卷积神经网络模型将MNIST未识别对的图片筛选出来
#添加保存图片的路径
image_path = './image_path/'
if os.path.exists(iamage_path) ==False:
            os.mkdir(image_path)

#查看训练好的模型是否存在，不存在则重新训练、
is_train_modle_exist = True

#查看训练好的模型是否存在
savePath = './mnist_conv/'
saveFile = save_path + 'mnist_conv.ckpt'
if os.path.exists(saveFile + '.index') == False:
            print ('Not found the CKPT files!')
            is_train_modle_exist == False

......
#导入保存的训练数据集
if is_train_modle_exist == True:
    saver.restore(sess,saveFile)
    print ("start testing...")
else:
    #初始化所有变量
    sess.run(tf.gloal_variables_intializer())
    for i in range (20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(acurracy,feed_dic={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d,training acurracy %g" % (i,train_acurracy))
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print ("end train ,start testing...")

if is_train_model_exist == False:
    saver.save(sess,saveFile)

#开始测试，将判断错的以JPG图片的形式保存
for i in range (mnist.test.labels.shape[0]):
    batch = mnist.test.next_batch(1)
    result = sess.run(correct_prediction,feed_dic:{x:batch[0],y_:batch[1],keep_prob:1.0})

    if result[0] == False:
        #查看机器把这张图片识别成什么数字的
        result1 = sess.run(y_conv,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        image = batch[0].reshape(28,28)
        #注意，这里想获取tf.argmax(batch[1],1))里的内容，一定要先用sess.run,只有run了才真正的运算
        filename = image_path + 'image_%d_%d_%d.jpg' % (i, sess.run(tf.argmax(batch[1],1))[0],
                                                        sess.run(tf.argmax(rsult1,1)))
        sm.toimage(image).save(filename)
        
