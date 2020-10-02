import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers

(xs, ys),_ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape)

xs = tf.convert_to_tensor(xs, dtype=tf.float32)/255.
db = tf.data.Dataset.from_tensor_slices((xs, ys))


for step, (x, y) in enumerate(db):
    print(step, x.shape, y, y.shape)

model = tf.keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])

#optimizer 最优控制器，帮助我们自动更新参数
optimizers = optimizers.SGD(learning_rate=0.001)

with tf.GradientTape() as tape:
    # [b, 28, 28] => [b, 784] 图像是二维的矩阵，加上图像数量这一维度，总共是三维。
    # 这里将图像的像素矩阵打平，变成一维的数组，完成一次降维
    x = tf.reshape(x, (-1, 28*28))
    # Step1  compute output
    # [b, 784] => [b, 10]
    out = model(x)
    # Step2  compute loss
    loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]