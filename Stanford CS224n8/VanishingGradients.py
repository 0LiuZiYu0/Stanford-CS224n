import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize']=(10.0,8.0)#设置画布的默认大小
# plt.rcParams['image.interpolation']='nearest'#设置插值的风格
# plt.rcParams['image.cmap']='gray'#设置画布的颜色
#
# np.random.seed(0)#保证后面seed(0)生成的随机数相同
# N=100 #每一个类别的数量
# D=2 #维度
# K=3 #类别数
# h=50
# h2=50
# X=np.zeros((N*K,D))
# num_train_examples=X.shape[0]
# y=np.zeros(N*K,dtype='uint8')
# for j in range(K):
#     ix=range(N*j,N*(j+1))
#     r=np.linspace(0.0,1,N) # 半径
#     t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2 # theta
#     X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]# 按行拼接两个矩阵
#     y[ix]=j
# fig=plt.figure()
# plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)#画点，c代表颜色，s代表点的大小
# plt.xlim([-1,1])#设置横坐标的范围大小
# plt.ylim([-1,1])#设置纵坐标的范围大小
# plt.show()

#sigmoid函数
def sigmoid(x):
    x=1/(1+np.exp(-x))
    return x
#sigmoid导函数
def sigmoid_grad(x):
    return x*(1-x)
#Relu激活函数
def relu(x):
    return np.maximum(0,x)

def three_layer_net(NONLINEARITY,X,y,model,step_size,reg):
    #参数初始化
    h=model['h']
    h2=model['h2']
    W1=model['W1']
    W2 = model['W2']
    W3 = model['W3']
    b1=model['b1']
    b2 = model['b2']
    b3 = model['b3']

    num_examples=X.shape[0]
    plot_array_1=[]
    plot_array_2=[]
    for i in range(50000):
        #前向传播
        if NONLINEARITY=='RELU':
            hidden_layer=relu(np.dot(X,W1)+b1)
            hidden_layer2=relu(np.dot(hidden_layer,W2)+b2)
            scores=np.dot(hidden_layer2,W3)+b3
        elif NONLINEARITY == 'SIGM':
            hidden_layer = sigmoid(np.dot(X, W1) + b1)
            hidden_layer2 = sigmoid(np.dot(hidden_layer, W2) + b2)
            scores = np.dot(hidden_layer2, W3) + b3
        exp_scores=np.exp(scores)
        probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)#softmax概率化得分[N*K]

        #计算损失
        correct_logprobs=-np.log(probs[range(num_examples),y])#整体的交叉熵损失
        data_loss=np.sum(correct_logprobs)/num_examples #平均交叉熵损失
        reg_loss=0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(W2*W2)+0.5*reg*np.sum(W3*W3)#正则化损失，W1*W1表示哈达玛积
        loss=data_loss+reg_loss
        if i%1000==0:
            print("iteration %d: loss %f"%(i,loss))

        #计算scores的梯度
        dscores=probs
        dscores[range(num_examples),y]-=1 #此处为交叉熵损失函数的求导结果，详细过程请看：https://blog.csdn.net/qian99/article/details/78046329
        dscores/=num_examples

        #反向传播过程
        dW3=(hidden_layer2.T).dot(dscores)
        db3=np.sum(dscores,axis=0,keepdims=True)


        if NONLINEARITY == 'RELU':
            # 采用RELU激活函数的反向传播过程
            dhidden2 = np.dot(dscores, W3.T)
            dhidden2[hidden_layer2 <= 0] = 0
            dW2 = np.dot(hidden_layer.T, dhidden2)
            plot_array_2.append(np.sum(np.abs(dW2)) / np.sum(np.abs(dW2.shape)))
            db2 = np.sum(dhidden2, axis=0)
            dhidden = np.dot(dhidden2, W2.T)
            dhidden[hidden_layer <= 0] = 0

        elif NONLINEARITY == 'SIGM':
            # 采用SIGM激活函数的反向传播过程
            dhidden2 = dscores.dot(W3.T) * sigmoid_grad(hidden_layer2)
            dW2 = (hidden_layer.T).dot(dhidden2)
            plot_array_2.append(np.sum(np.abs(dW2)) / np.sum(np.abs(dW2.shape)))
            db2 = np.sum(dhidden2, axis=0)
            dhidden = dhidden2.dot(W2.T) * sigmoid_grad(hidden_layer)

        dW1 = np.dot(X.T, dhidden)
        plot_array_1.append(np.sum(np.abs(dW1)) / np.sum(np.abs(dW1.shape)))#第一层的平均梯度记录下来
        db1 = np.sum(dhidden, axis=0)

        # 加入正则化得到的梯度
        dW3 += reg * W3
        dW2 += reg * W2
        dW1 += reg * W1

        # 记录梯度
        grads = {}
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        # 更新梯度
        W1 += -step_size * dW1
        b1 += -step_size * db1
        W2 += -step_size * dW2
        b2 += -step_size * db2
        W3 += -step_size * dW3
        b3 += -step_size * db3
    # 评估模型的准确度
    if NONLINEARITY == 'RELU':
        hidden_layer = relu(np.dot(X, W1) + b1)
        hidden_layer2 = relu(np.dot(hidden_layer, W2) + b2)
    elif NONLINEARITY == 'SIGM':
        hidden_layer = sigmoid(np.dot(X, W1) + b1)
        hidden_layer2 = sigmoid(np.dot(hidden_layer, W2) + b2)
    scores = np.dot(hidden_layer2, W3) + b3
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    # 返回梯度和参数
    return plot_array_1, plot_array_2, W1, W2, W3, b1, b2, b3
if __name__=='__main__':
    plt.rcParams['figure.figsize']=(10.0,8.0)#设置画布的默认大小
    plt.rcParams['image.interpolation']='nearest'#设置插值的风格
    plt.rcParams['image.cmap']='gray'#设置画布的颜色

    np.random.seed(0)#保证后面seed(0)生成的随机数相同
    N=100 #每一个类别的数量
    D=2 #维度
    K=3 #类别数
    h=50
    h2=50
    X=np.zeros((N*K,D))
    num_train_examples=X.shape[0]
    y=np.zeros(N*K,dtype='uint8')
    for j in range(K):
        ix=range(N*j,N*(j+1))
        r=np.linspace(0.0,1,N) # 半径
        t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2 # theta
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]# 按行拼接两个矩阵
        y[ix]=j
    fig=plt.figure()
    plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)#画点，c代表颜色，s代表点的大小
    plt.xlim([-1,1])#设置横坐标的范围大小
    plt.ylim([-1,1])#设置纵坐标的范围大小
    plt.show()

    #初始化模型参数
    h = 50
    h2 = 50
    model={}
    model['h'] = h # hidden layer 1 大小
    model['h2']= h2# hidden layer 2 大小
    model['W1']= 0.1 * np.random.randn(D,h)
    model['b1'] = np.zeros((1,h))
    model['W2'] = 0.1 * np.random.randn(h,h2)
    model['b2']= np.zeros((1,h2))
    model['W3'] = 0.1 * np.random.randn(h2,K)
    model['b3'] = np.zeros((1,K))
    Activation_Function='RELU'#选择激活函数  SIGM/RELU
    (plot_array_1, plot_array_2, W1, W2, W3, b1, b2, b3) = three_layer_net(Activation_Function, X, y, model,step_size=1e-1, reg=1e-3)

    #模型采用两种激活函数时，分别画出模型梯度的变化趋势

    plt.plot(np.array(plot_array_1))
    plt.plot(np.array(plot_array_2))
    plt.title('Sum of magnitudes of gradients -- hidden layer neurons')
    plt.legend((Activation_Function+" first layer", Activation_Function+" second layer"))
    plt.show()


