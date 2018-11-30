import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def create_data():
    #y=x*2+0.3*random(符合正态分布的采样)
    x_batch=np.linspace(-1,1,100)#将-1到1之间分成100段，组成输入数组
    y_batch=2*x_batch+np.random.randn(*x_batch.shape)*0.3#根据x产生y的真实值
    return x_batch,y_batch

def liner_regression():
    x=tf.placeholder(tf.float32,shape=(None,),name='x')#定义输入
    y=tf.placeholder(tf.float32,shape=(None,),name='y')#定义实际输出
    w=tf.Variable(np.random.normal(),name='w')#定义需训练的权重参数
    y_pred=tf.multiply(w,x)#定义预测输出
    loss=tf.reduce_mean(tf.square(y-y_pred))#计算loss
    return x,y,y_pred,loss

def run():
    x_batch,y_batch=create_data()#返回输入x和真实的y值
    x, y, y_pred, loss=liner_regression()#得到计算图中的节点
    optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#定义优化器
    init=tf.global_variables_initializer()#定义初始化参数操作
    with tf.Session() as session:#采用with定义session，因此会自动关闭session
        session.run(init)#初始化参数
        feed_dict={x:x_batch,y:y_batch}#定义计算图的输入字典
        for _ in range(50):#迭代次数
            loss_val=session.run(loss,feed_dict)#计算loss
            session.run(optimizer,feed_dict)#更新参数
            # loss_val,_=session.run([loss,optimizer],feed_dict)#也可以同时计算loss，更新参数
            print(loss_val)
        y_pred_batch=session.run(y_pred,feed_dict)#通过输入数据字典，获取计算图中计算出预测值
    plt.figure('liner_regression效果图')
    plt.scatter(x_batch,y_batch)#画散点图
    plt.plot(x_batch,y_pred_batch)#画连续图
    plt.show()

if __name__=='__main__':
    run()
