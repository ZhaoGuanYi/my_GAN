import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Reshape, Flatten
from keras.layers import LeakyReLU, BatchNormalization, Embedding, multiply

# 输入维度
img_rows = 28
img_cols = 28
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)

num_classes = 10
latent_dim = 100


def build_generrator():
    label = Input(shape=(1,), dtype='int32')
    noise = Input(shape=(latent_dim,))

    label_embedding = Flatten()(Embedding(10, latent_dim)(label))  # 10个种类的标签，映射到100维
    model_input = multiply([noise, label_embedding])  # 合并方法， 对应位置相乘

    net = Dense(256, activation=LeakyReLU(0.2))(model_input)
    net = BatchNormalization(momentum=0.8)(net)
    net = Dense(512, activation=LeakyReLU(0.2))(net)
    net = BatchNormalization(momentum=0.8)(net)
    net = Dense(1024, activation=LeakyReLU(0.2))(net)
    net = BatchNormalization(momentum=0.8)(net)
    net = Dense(np.prod(img_shape), activation='tanh')(net)
    img = Reshape(img_shape)(net)

    model = Model(inputs=[noise, label], outputs=img)
    model.summary()
    return model


def build_discriminator():
    img = Input(shape=img_shape)
    flat_img = Flatten()(img)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(10, np.prod(img_shape))(label))
    model_input = multiply([flat_img, label_embedding])

    net = Dense(512, activation=LeakyReLU(0.2))(model_input)
    net = Dense(512, activation=LeakyReLU(0.2))(net)
    net = Dropout(0.4)(net)
    net = Dense(512, activation=LeakyReLU(0.2))(net)
    net = Dropout(0.4)(net)
    out = Dense(1, activation='sigmoid')(net)

    model = Model(inputs=[img, label], outputs=out)
    model.summary()
    return model


'''
定义损失函数
'''
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_loss(d_real, d_fake):
    return bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)


# d_real与1比较，d_fake与0比较
def g_loss(generated_output):
    return bce(tf.ones_like(generated_output), generated_output)


def train():
    batch_size = 32

    epochs = 2000
    G = build_generrator()
    D = build_discriminator()

    (train_data, train_label), (_, _) = mnist.load_data()
    # train_data = train_data[:60000]
    # train_label = train_label[:60000]
    train_data = (train_data / 255) - 0.5
    train_data = np.expand_dims(train_data, axis=-1)
    train_label = train_label.reshape(-1, 1)

    optimizer = tf.keras.optimizers.Adam(1e-5)

    @tf.function
    def train_step(image, label):
        with tf.GradientTape(persistent=True) as tape:
            noise_vector = tf.random.normal(
                mean=0, stddev=1,
                shape=(image.shape[0], latent_dim))
            # 生成器样本
            fake_data = G([noise_vector, label])
            # 计算D_loss
            d_fake_data = D([fake_data, label])
            d_real_data = D([image, label])
            d_loss_value = d_loss(d_real_data, d_fake_data)
            # 计算G_loss
            g_loss_value = g_loss(d_fake_data)

        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        del tape
        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))

        return g_loss_value, d_loss_value, fake_data[0], label[0]

    for epoch in range(epochs):
        # flag = 1
        for t in range(train_data.shape[0] // batch_size):
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            image, label = train_data[idx], train_label[idx]

            g_loss_value, d_loss_value, generated, condition = train_step(image, label)
            print("epoch ", epoch, "step", t)
            print("g_loss:", g_loss_value.numpy(), "d_loss", d_loss_value.numpy())

            if epoch % 50 == 0 and t % 900 == 0:
                title = condition.numpy()
                plt.title(str(title))
                plt.imshow(tf.squeeze(generated).numpy(), cmap='gray')
                plt.show()
        if epoch % 10 == 0:
            G.save(r'./mnist/G_model/model' + str(epoch) + '.h5')
            D.save(r'./mnist/D_model/model' + str(epoch) + '.h5')


def test():
    filename = './mnist/G_model/model130.h5'
    G_model = tf.keras.models.load_model(filename)
    noise = tf.random.normal(
        mean=0, stddev=1,
        shape=(1, latent_dim))
    for i in range(10):
        label = tf.constant([i])
        image = G_model.predict([noise, label])

        plt.title(str(int(label)))
        plt.imshow(tf.squeeze(image).numpy(), cmap='gray')
        plt.show()


if __name__ == "__main__":
    # test()
    train()
