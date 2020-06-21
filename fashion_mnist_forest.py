from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow.examples.tutorials.mnist.input_data as input_data
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

def getAccuracy(y_test,y_pre):
    res=0
    for k,v in zip(y_test,y_pre):
        if k==v:
            res+=1
    return res/y_test.shape[0]

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load.get_data()
    #show_imgs(7, 7, x_train, y_train, class_name) #一样的展示功能

    print("start random forest")
    '''
    #遍历决策树数量，准确率基本在0.871-0.879，考虑计算结果，取150棵决策树
    for i in range(30,200,10):    
        clf_rf = RandomForestClassifier(n_estimators=i)    
        clf_rf.fit(x_train, y_train)     
        
        y_pred_rf = clf_rf.predict(x_test)    
        acc_rf = getAccuracy(y_test, y_pred_rf)  
        print("n_estimators = %d, random forest accuracy:%f" %(i,acc_rf))
    '''
    clf_rf = RandomForestClassifier(n_estimators=150)    
    #随机森林构建
    print(clf_rf)

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
        clf_rf.fit(x, y)
        end = time.time() #计算训练用时
        y_pred = clf_rf.predict(x_test)
        acy = getAccuracy(y_test,y_pred)
        acys.append(acy)
        times.append(end-start)

    for i in range(len(sizes)):
        print('对于随机森林分类，使用{size}个数据集时，训练所用时间{time}秒，得到准确率为{acy}'
            .format(size=sizes[i], time=times[i], acy=acys[i]))
