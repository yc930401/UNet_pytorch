import os
import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import PIL
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

training_image_dir = 'data/KITTI_SEMANTIC/Training_00/RGB/'
training_target_dir = 'data/KITTI_SEMANTIC/Training_00/GT/'
test_image_dir = 'data/KITTI_SEMANTIC/Validation_07/RGB/'
test_target_dir = 'data/KITTI_SEMANTIC/Validation_07/GT/'
output_dir = 'data/KITTI_SEMANTIC/Result/'
n_classes = 12
imgage_width = 800
image_height = 320

color_to_int = {(64, 0, 128): 0,
                (128, 128, 128): 1,
                (128, 128, 0): 2,
                (64, 64, 128): 3,
                (128, 0, 0): 4,
                (0, 0, 0): 5,
                (0, 0, 192): 6,
                (128, 64, 128): 7,
                (192, 128, 128): 8,
                (192, 192, 128): 9,
                (0, 128, 192): 10,
                (64, 64, 0): 11}

int_to_color = {0: (64, 0, 128),
                1: (128, 128, 128),
                2: (128, 128, 0),
                3: (64, 64, 128),
                4: (128, 0, 0),
                5: (0, 0, 0),
                6: (0, 0, 192),
                7: (128, 64, 128),
                8: (192, 128, 128),
                9: (192, 192, 128),
                10: (0, 128, 192),
                11: (64, 64, 0)}


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class KITTITrainingDataset(Dataset):

    def __init__(self, image_dir, target_dir, transform=None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.file_names = os.listdir(self.image_dir)
        self.color_idx = 0
        self.n_images = len(self.file_names)
        self.crop_box = (0, 0, imgage_width, image_height)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        training_image = np.array(Image.open(os.path.join(self.image_dir, self.file_names[idx])).crop(self.crop_box))
        target_image = np.array(Image.open(os.path.join(self.target_dir, self.file_names[idx])).crop(self.crop_box))
        img_height, img_width, channel = target_image.shape
        unique_colors = set(tuple(v) for m2d in target_image for v in m2d)
        for color in unique_colors:
            if color not in color_to_int.keys():
                color_to_int[color] = self.color_idx
                int_to_color[self.color_idx] = color
                self.color_idx += 1

        target_image = [[color_to_int[tuple(color)] for color in row] for row in target_image]
        if self.transform:
            training_image = self.transform(training_image)
        target_image = np.asarray([to_categorical(row, n_classes) for row in target_image])
        sample = {'image': training_image.reshape(3, img_height, img_width), \
                  'target': target_image.reshape(n_classes, img_height, img_width)}
        return sample


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) #padding=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) #padding=1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        return x


class Unet(nn.Module):
    def __init__(self, n_classes):
        super(Unet, self).__init__()
        self.n_classes = n_classes

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        self.conv6 = ConvBlock(768, 256)
        self.conv7 = ConvBlock(384, 128)
        self.conv8 = ConvBlock(192, 64)
        self.conv9 = ConvBlock(96, 32)

        self.conv10 = nn.Conv2d(32, n_classes, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        c3 = self.conv3(x)
        x = self.pool3(c3)
        c4 = self.conv4(x)
        x = self.pool4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        x = self.conv10(x)
        return x


def train(model, dataset, criterion, optimizer, batch_size=1, shuffle=False):
    optimizer.zero_grad()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    total_loss = []
    for i_batch, sample_batched in enumerate(dataloader):
        image, target = Variable(sample_batched['image'].cuda()), Variable(sample_batched['target'].cuda())
        image = image.type(torch.cuda.FloatTensor)
        topv_target, topi_target = target.type(torch.cuda.LongTensor).view(-1, n_classes).topk(1)
        target = topi_target.squeeze().detach()

        output = model(image).view(-1, n_classes)
        output = F.log_softmax(output, dim=-1) # or dim=1 also can

        loss = criterion(output, target)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(total_loss)


def evaluate(model, n_images):
    with torch.no_grad():
        dataset = KITTITrainingDataset(test_image_dir, test_target_dir)
        dataloader = DataLoader(dataset, batch_size=n_images, shuffle=False, num_workers=1)

        for i_batch, sample_batched in enumerate(dataloader):
            images, truths = Variable(sample_batched['image'].cuda()), Variable(sample_batched['target'].cuda())
            images = images.type(torch.cuda.FloatTensor)
            outputs = []
            targets = []

            for idx, output in enumerate(model(images)):
                target = truths[idx]
                topv_target, topi_target = target.type(torch.cuda.LongTensor).view(-1, n_classes).topk(1)
                target = topi_target.squeeze().detach()
                target = target.cpu().numpy()
                targets.append(target)

                # for prediction
                output = output.view(-1, n_classes)
                output = F.log_softmax(output, dim=-1)
                topv_output, topi_output = output.view(-1, n_classes).topk(1)
                output = topi_output.squeeze().detach()
                output = output.cpu().numpy()
                outputs.append(output)

                output_image = []
                target_image = []
                for i, item in enumerate(output):
                    output_image.append(list(int_to_color[item]))
                    target_image.append(list(int_to_color[target[i]]))
                output_image = np.array(output_image, dtype='uint8').reshape(image_height, imgage_width, 3) # int32 by defaulr
                target_image = np.array(target_image, dtype='uint8').reshape(image_height, imgage_width, 3)
                original_image = np.array(images[idx], dtype='uint8').reshape(image_height, imgage_width, 3)
                img_output = Image.fromarray(output_image, 'RGB')
                img_target = Image.fromarray(target_image, 'RGB')
                img_original = Image.fromarray(original_image, 'RGB')
                img_output.save(os.path.join(output_dir, 'output_' + dataset.file_names[idx]))
                img_target.save(os.path.join(output_dir, 'target_' + dataset.file_names[idx]))
                img_original.save(os.path.join(output_dir, 'original_' + dataset.file_names[idx]))

            matrix = confusion_matrix(np.array(targets)[0], np.array(outputs)[0])
            FP = matrix.sum(axis=0) - np.diag(matrix)
            FN = matrix.sum(axis=1) - np.diag(matrix)
            TP = np.diag(matrix)
            TN = matrix.sum(axis=0).sum() - (FP + FN + TP)
            IoU = np.asarray(TP / (TP + FP + FN))

            return IoU.mean()


def trainIters(model, learning_rate, weight_decay, batch_size, n_epochs=1):
    dataset = KITTITrainingDataset(training_image_dir, training_target_dir)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    for i in range(n_epochs):
        loss = train(model, dataset, criterion, optimizer, batch_size)
        torch.save(model, 'model/model.pkl')
        print('Iter: {}, Loss: {}'.format(i, loss))



if __name__ == '__main__':
    try:
        model = torch.load('model/model.pkl')
    except:
        model = Unet(n_classes).to(device)
    # Train a model
    trainIters(model, learning_rate=0.000001, weight_decay=0.0001, batch_size=1, n_epochs=20)

    # Test a model
    IoU = evaluate(model, n_images=3)
    print('IoU: ', IoU)

