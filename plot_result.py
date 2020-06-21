import matplotlib.pyplot as plt

'''
    对三种模型的结果进行绘图处理，但考虑全部运行一次
过于麻烦。所以这里直接用数据了。结果可以在result文件
夹下看到
'''

sizes = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
#SVM
acy_svm = [0.773, 0.8441, 0.8581, 0.872, 0.8772, 0.8828]
time_svm = [0.4317, 11.7188, 53.325, 170.2908, 345.9, 572.6873]
#Random forest
acy_rf = [0.783, 0.8425, 0.8562, 0.8679, 0.8744, 0.8758]
time_rf = [0.7375, 8.6342, 24.7495, 55.9796, 88.3003, 122.8512]
#deep learning
acy_dl = [0.7622, 0.8411, 0.8564, 0.8765, 0.8875, 0.8948]
time_dl = [2.63, 4.5417, 11.4254, 22.3911, 34.0436, 45.1231]
#画准确度
plt.figure()
plt.plot(sizes, acy_svm, "r-", linewidth=2, label='SVM')
plt.plot(sizes, acy_rf, "g--", linewidth=2, label='Random Forest')
plt.plot(sizes, acy_dl, "b-.", linewidth=2, label='Deep Learn')
plt.grid(True)
plt.legend(loc="center right", fontsize=10)
plt.title("The accuracy of the three models", fontsize=14)
plt.xlabel("size of data")
plt.ylabel("accuracy")
plt.axis([0, 1 , 0.75, 0.95])
plt.show()
#画时间
plt.figure()
plt.plot(sizes, time_svm, "r-", linewidth=2, label='SVM')
plt.plot(sizes, time_rf, "g--", linewidth=2, label='Random Forest')
plt.plot(sizes, time_dl, "b-.", linewidth=2, label='Deep Learn')
plt.grid(True)
plt.legend(loc="center right", fontsize=10)
plt.title("The train-time of the three models", fontsize=14)
plt.xlabel("size of data")
plt.ylabel("used time/s")
plt.axis([0, 1 , 0, 600])
plt.show()