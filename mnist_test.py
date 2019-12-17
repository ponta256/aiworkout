from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--saved-model', default='mnist.pth', help='Saved Model')
    parser.add_argument('--invert', action='store_true',
                        default=False, help='Invert Image') 
    parser.add_argument(
        "input_file",
        help="Input image, list, directory, or npy."
    )

    args = parser.parse_args()

    transform =  transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])

    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(args.saved_model))
    model.eval()
    with open(args.input_file, 'rb') as f:
        image = Image.open(f)
        if args.invert:
            image = ImageOps.invert(image)
        image = image.convert('L')
        cv2.imshow('', np.array(image, dtype=np.uint8))
        cv2.waitKey()
        image = transform(image)
        image = image.unsqueeze(0)
        output = model(image)
    pred = output.argmax()
    print('Prediction: ', int(pred))

    
if __name__ == '__main__':
    main()
