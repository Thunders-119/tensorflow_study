from tensorflow import keras
import picture_classify as pc

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '橡皮底帆布鞋', '包', '高帮靴']
    train_images, test_images = pc.picture_make1(train_images,test_images)
    model = keras.models.load_model('My_first_model.h5')
    pc.test_model(model, test_images, test_labels)
    predictions = pc.calculate(model, test_images)
    # print(predictions[93], '\n', class_names[np.argmax(predictions[2])])
    model_show1 = pc.model_show(predictions, 93, test_images, test_labels, class_names)
    model_show1.calculate_show2()
