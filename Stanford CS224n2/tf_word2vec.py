from numpy import random
import numpy as np
import collections
import math
import tensorflow as tf
import jieba
from collections import Counter


# 此函数作用是对初始语料进行分词处理
def cut_txt(old_file):
    cut_file = old_file + '_cut.txt' # 分词之后保存的文件名
    fi = open(old_file, 'r', encoding='utf-8') #注意操作前要把文件转化成utf-8文件
    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 采用精确模式切词
    str_out = ' '.join(new_text)

    #去除停用词
    stopwords = [line.strip() for line in open('DataSet/中文停用词.txt', 'r',encoding='utf-8').readlines()]
    for stopword in stopwords:
        str_out=str_out.replace(' '+stopword+' ',' ')
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)
    ret_list=str_out.split()#训练语料
    ret_set=list(set(ret_list))#字典
    list_len=len(ret_list)
    set_len=len(set(ret_list))
    print('总字数 总词数 :')
    print(list_len,set_len) #总字数 总词数
    print('词频字典 :')
    print(dict(Counter(ret_list)))# 词频字典
    return ret_list,ret_set

# 预处理切词后的数据
def build_dataset(words, n_words):
    count = [['UNK', -1]] #存放词频做大的n_words个单词，第一个元素为单词，第二个元素为词频。UNK为其他单词
    count.extend(collections.Counter(words).most_common(n_words - 1))#获取词频做大的n_words-1个单词（因为有了UNK，所以少一个）
    dictionary = dict() #建立单词的索引字典
    for word, _ in count:
        dictionary[word] = len(dictionary) #建立单词的索引字典，key为单词，value为索引值
    data = list()# 建立存放训练语料对应索引号的list，也就是说将训练的语料换成每个单词对应的索引
    unk_count = 0 #计数UNK的个数
    for word in words:
        if word in dictionary:
            index = dictionary[word]#获取单词在字典中的索引
        else:
            index = 0 #字典中不存在的单词（UNK），在字典中的索引是0
            unk_count += 1 #每次遇到字典中不存在的单词 UNK计数器加1
        data.append(index)#将训练语料对应的索引号存入data中
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))#将单词的索引字典反转，即将key和value对换
    return data, count, dictionary, reversed_dictionary#训练语料的索引；top大的单词词频；字典{单词：索引值}；字典{索引值：单词}


# 为 skip-gram model 产生bathch训练样本.
#从文本总体的第skip_window+1个单词开始，每个单词依次作为输入，它的输出可以是上下文范围内的单词中的任何一个单词。一般不是取全部而是随机取其中的几组，以增加随机性。
def generate_batch(batch_size, num_skips, skip_window):#batch_size 就是每次训练用多少数据，skip_window是确定取一个词周边多远的词来训练，num_skips是对于一个输入数据，随机取其窗口内几个单词作为上下文（即输出标签）。
  global data_index
  assert batch_size % num_skips == 0#保证batch_size是 num_skips的整倍数，控制下面的循环次数
  assert num_skips <= 2 * skip_window #保证num_skips不会超过当前输入的的上下文的总个数
  batch = np.ndarray(shape=(batch_size), dtype=np.int32) #存储训练语料中心词的索引
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)#存储训练语料中心词对应的上下文的索引
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  #这个很重要，最大长度是span，后面如果数据超过这个长度，前面的会被挤掉，这样使得buffer里面永远是data_index周围的span歌数据，
  #而buffer的第skip_window个数据永远是当前处理循环里的输入数据
  buffer = collections.deque(maxlen=span)#一个完整的窗口存储器
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span #获取下一个要进入队列的训练数据的索引
  for i in range(batch_size // num_skips):#一个batch一共需要batch个训练单词对，每个span中随机选取num_skips个单词对，所以要循环batch_size // num_skips次
    target = skip_window  # 中心词索引在buffer中的位置
    targets_to_avoid = [skip_window] #自己肯定要排除掉，不能自己作为自己的上下文
    for j in range(num_skips):#采样num_skips次
      while target in targets_to_avoid:
        target = random.randint(0, span - 1) #随机取一个，增强随机性，减少训练时进入局部最优解
      targets_to_avoid.append(target)#采样到某一个上下文单词后，下一次将不会再采样
      batch[i * num_skips + j] = buffer[skip_window] #这里保存的是训练的输入序列
      labels[i * num_skips + j, 0] = buffer[target] #这里保存的是训练时的输出序列，也就是标签
    if data_index == len(data): #超长时回到开始
        buffer.extend(data[0:span])
        data_index = span
    else:
        buffer.append(data[data_index]) #append时会把queue的开始的一个挤掉
        data_index += 1 #此处是控制模型窗口是一步步往后移动的
  data_index = (data_index + len(data) - span) % len(data)# 倒回一个span，防止遗漏最后的一些单词
  return batch, labels#返回 输入序列  输出序列


############ 第一步：对初始语料进行分词处理 ############
train_data,dict_data=cut_txt('DataSet/倚天屠龙记.txt')#切词
vocabulary_size =10000#字典的大小，只取词频top10000的单词

############ 第二步：预处理切词后的数据 ############
data, count, dictionary, reverse_dictionary = build_dataset(train_data,vocabulary_size)#预处理数据
print()
print(data)
print()
print(count)
print()
print(dictionary)
print()
print(reverse_dictionary)
print('Most common words (+UNK)', count[:5])#词频最高的前5个单词
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])#前10个训练数据索引及其具体单词

############ 第三步：为 skip-gram model 产生bathch训练样本. ############
data_index = 0#控制窗口滑动的
batch, labels = generate_batch(batch_size=128, num_skips=8, skip_window=5)#产生一个batch的训练数据。batch大小128；从上下文中随机抽取8个单词作为输出标签；窗口大小5（即一个窗口下11个单词，1个人中心词，10个上下文单词）；
for i in range(10):#输出一下一个batch中的前10个训练数据对（即10个训练样本）
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

############ 第四步： 构造一个训练skip-gram 的模型 ############
batch_size = 128 #一次更新参数所需的单词对
embedding_size = 128  # 训练后词向量的维度
skip_window = 5  #窗口的大小
num_skips = 8  # 一个完整的窗口（span）下，随机取num_skips个单词对（训练样本）

# 构造验证集的超参数
valid_size = 16  # 随机选取valid_size个单词，并计算与其最相似的单词
valid_window = 100  # 从词频最大的valid_window个单词中选取valid_size个单词
valid_examples = np.random.choice(valid_window, valid_size, replace=False)#选取验证集的单词索引
num_sampled = 64  #负采样的数目

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])  #中心词
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) #上下文
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32) #验证集

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))#定义单词的embedding
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)#窗口查询中心词对应的embedding

    # 为 NCE loss构造变量
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))#权重
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))#偏差

    # 对于一个batch，计算其平均的 NEC loss
    # 采用负采样优化训练过程
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,inputs=embed,num_sampled=num_sampled,num_classes=vocabulary_size))

    #采用随机梯度下降优化损失函数，学习率采用1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 从字典中所有的单词计算一次与验证集最相似（余弦相似度判断标准）的单词
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))#计算模
    normalized_embeddings = embeddings / norm #向量除以其模大小，变成单位向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)#选出验证集的单位向量
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)#验证集的单位向量，乘以所有单词的单位向量。得到余弦相似度

    # 变量初始化
    init = tf.global_variables_initializer()

############ 第五步：开始训练 ############
num_steps = 100001 #迭代次数
with tf.Session(graph=graph) as session:
    init.run()
    print('开始训练')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)#产生一个batch
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}#tensor的输入


        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)#得到一个batch的损失值
        average_loss += loss_val #损失值累加

        if step % 2000 == 0:#每迭代2000次，就计算一次平均损失，并输出
            if step > 0:
                average_loss /= 2000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0 #每2000次迭代后，将累加的损失值归零

        if step % 10000 == 0:#每迭代10000次 就计算一次与验证集最相似的单词，由于计算量很大，所以尽量计算相似度
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]#得到需验证的单词
                top_k = 8  # 和验证集最相似的top_k个单词
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]#最邻近的单词的索引，[1:top_k + 1]从1开始，是跳过了本身
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]#获得第k个最近的单词
                    log_str = '%s %s,' % (log_str, close_word) #拼接要输出的字符串
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
