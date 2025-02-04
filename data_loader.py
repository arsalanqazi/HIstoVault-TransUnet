from libs import *


"""This Dataloader is used by us by it is a generic code for using a data loader 
anyone who wanna load the data can get an idea and then write their own code 
as per there directory structure to load the data and utilise it."""

def is_jpg_file(filename):
    return filename.lower().endswith('.jpg')

class DataLoader:
    def __init__(self, data_dir, img_size=(224, 224), test_size=0.2, val_size=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size

    def load_and_balance_data(self):
        normal_images, cancer_images = [], []
        
        # Load normal images
        normal_dir = os.path.join(self.data_dir, 'normal')
        for img_name in os.listdir(normal_dir):
            img_path = os.path.join(normal_dir, img_name)
            if is_jpg_file(img_name):
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.img_size)
                normal_images.append(img)

        # Load cancer images
        cancer_dir = os.path.join(self.data_dir, 'cancer')
        for img_name in os.listdir(cancer_dir):
            img_path = os.path.join(cancer_dir, img_name)
            if is_jpg_file(img_name):
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.img_size)
                cancer_images.append(img)

        # Balance the datasets
        min_samples = min(len(normal_images), len(cancer_images))
        normal_images, cancer_images = normal_images[:min_samples], cancer_images[:min_samples]

        # Create labels
        normal_labels = [0] * len(normal_images)
        cancer_labels = [1] * len(cancer_images)

        # Combine and shuffle
        images = np.array(normal_images + cancer_images)
        labels = np.array(normal_labels + cancer_labels)
        indices = np.random.permutation(len(images))

        return images[indices], labels[indices]

    def split_data(self, images, labels):
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=self.test_size, stratify=labels
        )
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=self.val_size, stratify=train_labels
        )

        return train_images, train_labels, test_images, test_labels, val_images, val_labels

    def load_data(self):
        images, labels = self.load_and_balance_data()
        return self.split_data(images, labels)
