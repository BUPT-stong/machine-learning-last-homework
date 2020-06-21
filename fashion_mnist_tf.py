#这部分代码参考了很多课上内容
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.model_selection import train_test_split

import fashion_mnist_load as load #加载fashion-mnist数据集

#fashion-mnist数据集中的分类
class_name={0 : 'T-shirt/top',
            1 : 'Trouser',
            2 : 'Pullover',
            3 : 'Dress',            
            4 : 'Coat',            
            5 : 'Sandal',            
            6 : 'Shirt',            
            7 : 'Sneaker',            
            8 : 'Bag',            
            9 : 'Ankle boot'}     
#数据可视化
def show_imgs(n_rows,n_cols,x_data,y_data,class_name):    
    assert len(x_data)==len(y_data)    
    plt.figure(figsize=(n_cols*1.4,n_rows*1.6))    
    for row in range(n_rows):        
        for col in range(n_cols):            
            index=n_cols*row+col            
            plt.subplot(n_rows,n_cols,index+1)            
            plt.imshow(x_data[index].reshape(28,28),cmap="binary",interpolation='nearest')            
            plt.axis('off')            
            plt.title(class_name[y_data[index]])                
    plt.show() 
    return None

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load.get_data()
    #show_imgs(7, 7, x_train, y_train, class_name) #一样的展示功能
    
    #不一样的是，神经网络进行浮点数计算，这里进行归一化
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32")/255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32")/255

    #建立模型
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=[28, 28, 1]),    
            keras.layers.Dense(300, activation=tf.nn.relu), 
            keras.layers.Dense(100, activation=tf.nn.relu), 
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

    #编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    #训练模型
    sizes = [0.01, 0.1, 0.25, 0.5, 0.75, 1] #决定使用数据量
    times = []
    acys = []
    for size in sizes:
        if size != 1:
            x, a, y, b = train_test_split(x_train, y_train 
                                        ,test_size=1-size,random_state=5)
            #划分数据集，使用相同随机种子划分
        else:
            x = x_train
            y = y_train
        
        start = time.time()
        model.fit(x, y, epochs=10) #学习10次就有很好效果
        end = time.time() #计算训练用时
        test_loss, test_acy = model.evaluate(x_test, y_test)
        acys.append(test_acy)
        times.append(end-start)
    
    for i in range(len(sizes)):
        print('对于两层神经网络分类，使用{size}个数据集时，训练所用时间{time}秒，得到准确率为{acy}'
            .format(size=sizes[i], time=times[i], acy=acys[i]))
    
    '''
    #下面是调用模型的方法
    predictions = model.predict(x_test)
    print(class_name[predictions[0]])
    '''