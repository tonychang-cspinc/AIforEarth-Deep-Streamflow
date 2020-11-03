import tensorflow as tf

class SimpleModel(tf.keras.Model):
    def __init__(self, **kwargs):
        self.num_classes = num_classes
        self.input_shape = input_shape
        super(SimpleModel, self).__init__(**kwargs)
        self.input = tf.keras.layers.Conv2D(16,3,\
                padding='same',\
                activation='relu',\
                input_shape=self.input_shape)
        self.conv1 = tf.keras.layers.Conv2D(32, 3,\
                padding='same',\
                activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3,\
                padding='same',\
                activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(128,\
                activation='relu')
        self.output = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs, training=False):
        x = self.input(inputs)
        #if training:
        #    x = self.dropout(x, training=training)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output(x)





