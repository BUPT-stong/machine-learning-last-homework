import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action="ignore")

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
    show_imgs(7, 7, x_train, y_train, class_name) #展示功能
    
    #不一样的是，神经网络进行浮点数计算，这里进行归一化
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32")/255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32")/255

    #建立模型
    model = keras.Sequential([
    #(-1,28,28,1)->(-1,28,28,32)
    keras.layers.Conv2D(input_shape=(28, 28, 1),filters=32,kernel_size=5,strides=1,padding='same'),     # Padding method),
    #(-1,28,28,32)->(-1,14,14,32)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,14,14,32)->(-1,14,14,64)
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,14,14,64)->(-1,7,7,64)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,7,7,64)->(-1,7*7*64)
    keras.layers.Flatten(),
    #(-1,7*7*64)->(-1,256)
    keras.layers.Dense(256, activation=tf.nn.relu),
    #(-1,256)->(-1,10)
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    print(model.summary()) #展示模型

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
        print('对于卷积神经网络分类，训练10次，使用{size}个数据集时，训练所用时间{time}秒，得到准确率为{acy}'
            .format(size=sizes[i], time=times[i], acy=acys[i]))

    '''
    #保存模型,model文件夹下可看
    pd.DataFrame(save.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    model.save_weights("./model/fashion_model.ckpt")
    '''
    
    #下面是调用模型的方法,预测测试集前10图片
    pred = np.argmax(model.predict(x_test[:10]),1)
    test_label = y_test[:10]
    for i in range(len(pred)):
        print('预测第{i}张图片为{pred}类别，实际上它是{test}类别'
                .format(i=i, pred=class_name[pred[i]], test=class_name[test_label[i]]))
    