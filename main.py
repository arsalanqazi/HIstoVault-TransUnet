from data_loader import DataLoader
from trainer import Trainer
from model import TransUnet
from libs import tf

# Set data directory
data_dir = 'Director/path'
data_loader = DataLoader(data_dir)

# Load data
train_images, train_labels, test_images, test_labels, val_images, val_labels = data_loader.load_data()

# Model parameters
input_shape = (224, 224, 3)
num_classes = 2 # You can change it as per the requirement but you have to change the loss function and the output layer of the model as well

# Define and compile model
transunet = TransUnet(input_shape, num_classes, weight_decay=0)
strategy = tf.distribute.MirroredStrategy()  # For multi-GPU training

with strategy.scope():
    model = transunet.build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train and evaluate model
trainer = Trainer(model, train_images, train_labels, test_images, test_labels, val_images, val_labels)
trainer.train()
trainer.evaluate()
trainer.plot_metrics()