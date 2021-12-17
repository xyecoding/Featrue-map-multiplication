from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from imutils import paths
import cv2
import os
image_paths = list(paths.list_images('/media/new_2t/yexiang/.torch/datasets/caltech101/101_ObjectCategories'))
data = []
labels = []
label_names = []
for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    if label == 'BACKGROUND_Google':
        continue

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data.append(image)
    label_names.append(label)
    labels.append(label)

print(len(labels))
data = np.array(data)
labels = np.array(labels)

lb = LabelEncoder()
#lb = LabelBinarizer()
labels = lb.fit_transform(labels)
#labels = lb.fit_transform(labels)

print(len(lb.classes_))
(x_train, x_val , y_train, y_val) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

#(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=42)

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize((64, 64)),
    #  transforms.RandomRotation((-30, 30)),
    #  transforms.RandomHorizontalFlip(p=0.5),
    #  transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
	 transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = ImageDataset(x_train, y_train, train_transform)
val_data = ImageDataset(x_val, y_val, val_transform)
#test_data = ImageDataset(x_test, y_test, val_transform)

# dataloaders
#trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
#valloader = DataLoader(val_data, batch_size=16, shuffle=True)
#testloader = DataLoader(test_data, batch_size=16, shuffle=False)


