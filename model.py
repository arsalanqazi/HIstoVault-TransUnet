from libs import *

class TransUnet:
    def __init__(self, input_shape, num_classes, num_layers=4, num_heads=8, ff_dim=512, weight_decay=2e-3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.weight_decay = weight_decay

    def transformer_block(self, inputs):
        attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
        attention_output = layers.Dropout(0.2)(attention_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        ffn = layers.Dense(self.ff_dim, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(out1)
        ffn = layers.Dense(inputs.shape[-1], kernel_regularizer=regularizers.l2(self.weight_decay))(ffn)
        ffn_output = layers.Dropout(0.2)(ffn)
        return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        skip_connections = [base_model.get_layer(name).output for name in ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']]
        encoder_output = base_model.get_layer('conv5_block3_out').output

        patches = layers.Reshape((encoder_output.shape[1] * encoder_output.shape[2], encoder_output.shape[-1]))(encoder_output)

        for _ in range(self.num_layers):
            patches = self.transformer_block(patches)

        transformer_output = layers.Reshape(encoder_output.shape[1:])(patches)

        x = layers.UpSampling2D((2, 2))(transformer_output)
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip_connections[3]])

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip_connections[2]])

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip_connections[1]])

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip_connections[0]])

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(self.weight_decay))(x)

        model = Model(inputs, outputs)
        return model
