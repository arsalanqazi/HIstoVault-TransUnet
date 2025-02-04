from libs import *
class Trainer:
    def __init__(self, model, train_images, train_labels, test_images, test_labels, val_images, val_labels, batch_size=32, epochs=100):
        self.model = model
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None

    def train(self):
        # Define the learning rate scheduler callback
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.history = self.model.fit(
            self.train_images, self.train_labels,
            validation_data=(self.val_images, self.val_labels),
            epochs=self.epochs,
            batch_size=self.batch_size,
            #callbacks=[self.lr_scheduler]
        )
        
    def lr_schedule(self, epoch, lr):
        decay_rate = 0.1
        decay_step = 8
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_images, self.test_labels)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    def plot_metrics(self):
       
        # Plot training & validation accuracy values
        plt.figure(figsize=(14, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        plt.show()
