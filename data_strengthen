#tensorflow数据增强实现
import tensorflow as tf
import matplotlib.pyplot as plt

#读取原始图像数据。转换成二进制串
image_data = tf.gfile.FastGFile('la.jpg','r').read()


with tf.Session() as sess:
    #对图像使用jpg格式解码,得到三维数据
    pltdata = tf.image.decode_jpeg(image_data)

    #对图像上下反转
    pltdata = tf.image.flip_up_down(pltdata)

    #将数据类型转化成uint8
    pltdata = tf.image.convert_image_dtype(pltdata,dtype = tf.uint8)

    #调整图片大小resize,提供有4种方法
    resize_img = tf.image.resize_images(pltdata,[300,300],method = 0)

    #图片剪切填充，放大图片就自动周围填充黑色，缩小图片就从图中间剪切
    resize_img = tf.iamge.resize_image_with_crop_or_pad(pltdata,600,600)

    #按比例大小缩小
    central_cropped = tf.iamge.central_crop(pltdata,0.5)

    #方框剪切,(50,50)指高、宽在图片在左上角的偏移量
    resize_img = tf.image.pad_to_bounding_box(pltdata,50,50,600,600)
    resize_img = tf.image.crop_to_bounding_box(pltdata,50,50,600,600)

    #翻转以及随机翻转：上下翻转、左右翻转、对角线翻转
    '''
    tf.image.flip_up_down(pltdata)
    tf.image.flip_left_right(pltdata)
    tf.image.transpose_image(pltdata)
    '''
    #随机翻转：随机上下左右、亮度、对比度、色相、饱和度
    '''
    tf.image.random_flip_up_down()
    tf.iamge.random_left_right()
    tf.image.random_brightness()
    tf.image.contrast()
    tf.image.random_hue()
    tf.image.random_saturation()
    '''

    #图像色彩调整
    '''
    tf.image.adjust_contrast()
    tf.image.adjust_gamma()
    tf.image.adjust_hue()
    tf.image.adjust_sturation()
    tf.image.adjust_brightness()
    tf.image.per_image_standardization()
    '''

    #图片标注
    #tf.iamge.draw_bounding_bosxes输入一个batch的数据
    #也就是多个图片组成的四维矩阵,第一个参数应为实数tf.float32
    batched = tf.expand_dims(pltdata,0)
    #x给出图像相对位置[y_min,x_min,y_max,x_max]
    boxes = tf.constant([[[0.2,0.3,0.5,0.8]]])
    res = tf.image.draw_bounding_boxes(batched,boxes,name = 'bounding_box')
    plt.subplot(121),plt.imshow(pltdata.eval()),plt.title('original')
    plt.subplot(122),plt.imshow(np.asarray(res.eval())[0]),plt.title('result')
    plt.imsave(fname = "save.jpg",arr = np.asarray(res.eval()[0])#保存图片

    #截取标记部分
    '''
    随机截取图像上有信息含量的部分，也可以提高模型健壮性
    此函数为图像生成单个随机变形的边界框，函数输出可用于原始图像裁剪的单个边界框
    返回值为3个张量：begin,size,bboxes
    前两个用于tf.slice剪裁图像
    后者可以用tf.image.draw_bounding_boxes函数画出边界框
    boxes = tf.constant([[[0.2,0.3,0.5,0.8]]])
    print(np.asarray(pltdata).shape)
    begin,size,bbox_for_draw =
    tf.bounding.sample_distorted_bounding_box(tf.shape(pltdata),
                                bounding_boxes = boxes,
                                min_object_covered=0.1)
    #batched = tf.expand_dims(tf.image.convert_image_dtype(pltdata,tf.float32),0)
    #image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
    distorted_image = tf.slice(pltdata,begin,size)
    '''
    
    #显示图像
    plt.imshow(pltdata.eval())#RGB模式输出一个三维数组
    plt.imshow(resize_img.eval())
    plt.imshow(central_cropped())
    plt.imshow(distorted_image.eval())
    plt.show()

    
