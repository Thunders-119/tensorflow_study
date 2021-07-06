# -*-  coding:utf-8  -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

def test_picture(train_images,train_labels,test_images,test_labels):
    '''
    查看数据集属性
    '''
    print('-------------------')
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    print('-------------------')
def picture_show(train_images):
    '''
    查看数据集中的任意图片属性
    '''
    plt.figure()
    plt.imshow(train_images[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()
def picture_make1(train_images,test_images):
    '''
    处理图片并返回新图片
    '''
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images
def picture_show2(imgs,labels):
    '''
    查看各类别的图片
    '''
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()
def test1(train_images, train_labels):
    '''
    训练模型
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    return model
def test_model(model, test_images, test_labels):
    '''
    测试模型并返回模型准确率
    '''
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\n模型准确率:'+str(test_acc)+'%')
def calculate(model,test_images):
    '''
    使用模型对测试集进行预测,返回预测样本的分类集合
    预测结果是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”
    '''
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    return predictions
class model_show:
    predictions = None
    i = None
    test_images = None
    test_labels = None
    class_names = None
    def __init__(self,  predictions, i, test_images, test_labels, class_names):
        self.predictions = predictions
        self.i = i
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names
    def plot_image(self,i, predictions_array, true_label, img, class_names):
        '''
        展示待预测图像
        '''
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)
    def plot_value_array(self,i, predictions_array, true_label, class_names):
        '''
        展示预测分类
        '''
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10), class_names, rotation=45)
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#666666")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
    def calculate_show2(self):
        '''
        对一些图像进行预测
        '''
        # i = 0
        i = self.i
        predictions = self.predictions
        test_images = self.test_images
        test_labels = self.test_labels
        class_names = self.class_names
        num_rows = 3
        num_cols = 2
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 3 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions[i], test_labels, test_images, class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions[i], test_labels, class_names)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # print(tf.__version__)
    # test_picture(train_images, train_labels, test_images, test_labels)
    # picture_show(train_images)
    train_images, test_images = picture_make1(train_images,test_images)
    # picture_show(train_images)
    # picture_show2(train_images,train_labels)
    model = test1(train_images, train_labels)
    test_model(model, test_images, test_labels)
    predictions = calculate(model, test_images)
    # print(predictions[2], '\n', class_names[np.argmax(predictions[2])])
    model_show1= model_show(predictions, 26, test_images, test_labels, class_names)
    model_show1.calculate_show2()
    # model.save('My_first_model.h5')