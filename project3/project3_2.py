#torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim

#for read file
import csv
import json
import os
from PIL import Image
from turtle import forward
from tqdm import tqdm

class res_block(nn.Module):
    def __init__(self, in_, out_, stride=1):
        super().__init__()

        #bottle neck layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_, out_, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_), nn.ReLU()
        )
        #convolution layer(3*3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_, out_, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_), nn.ReLU()
        )
        #bottle neck and extension depth
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_, out_ * 2, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_ * 2)
        )
        #identity mapping
        self.shortway = nn.Sequential(
            nn.Conv2d(in_, out_ * 2, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ * 2)
        ) if (stride != 1 or in_ != out_ * 2) else nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        output = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += self.shortway(output)
        x = self.relu(x)
        return x

class MyModel(nn.Module) :
    def __init__(self, res_block):
        super().__init__()
        #first convolve layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        #rest of residual block
        #depth 64 -> 128 -> 128
        self.conv2 = nn.Sequential(
            res_block(64, 64, 1),
            res_block(128, 64, 1)
        )
        #depth 128 -> 256 -> 256 -> 256
        self.conv3 = nn.Sequential(
            res_block(128, 128, 2),
            res_block(256, 128, 1),
            res_block(256, 128, 1)
        ) 
        #depth 256 -> 512 -> 512 -> 512 -> 512 -> 512
        self.conv4 = nn.Sequential(
            res_block(256, 256, 2),
            res_block(512, 256, 1),
            res_block(512, 256, 1),
            res_block(512, 256, 1),
            res_block(512, 256, 1)
        ) 
        #depth 512 -> 1024 -> 1024 -> 1024
        self.conv5 = nn.Sequential(
            res_block(512, 512, 2),
            res_block(1024, 512, 1),
            res_block(1024, 512, 1)
        ) 
        #avg pooling 1*1*1024
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        #fully connected layer
        self.fc = nn.Linear(1024, 80)

    #forward function
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MyDataset(Dataset):

    def __init__(self,meta_path,root_dir,transform=None) :
        super().__init__()
        self.img_labels = []
        with open(meta_path, 'r') as f:
            json_data = json.load(f)
        for l in json_data['annotations']:
            elemt = []
            elemt.append(l['file_name'])
            elemt.append(l['category'])
            self.img_labels.append(elemt)
        self.img_dir = root_dir
        self.transform = transform
    
    #number of image
    def __len__(self):
        return len(self.img_labels)

    #return image data and label
    def __getitem__(self,idx) :
        label = int(self.img_labels[idx][1])
        path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = Image.open(path).convert('RGB')
        if self.transform: image = self.transform(img)
        return image, label

#for test data set
class MyDatasett(Dataset):
    def __init__(self,meta_path,root_dir,transform=None) :
        super().__init__()
        self.img_dir = root_dir
        self.transform = transform
        self.file_name = os.listdir(root_dir)

    #number of image
    def __len__(self):
        return len(self.file_name)

    #return image data and file name
    def __getitem__(self,idx) :
        path = os.path.join(self.img_dir, self.file_name[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image, self.file_name[idx]

def train(epochs) :
    model_name = MyModel
    checkpoint_path = './drive/MyDrive/model.pth'
    mode = 'train' 
    data_dir = "./drive/MyDrive/train_data"
    meta_path = "./drive/MyDrive/answer.json"
    model = get_model(model_name,checkpoint_path)

    #train transforms
    transt = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(90, expand=False)])
    #test trans forms
    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    data_transforms = {
        'train' : transt, 
        'test' : trans
    }

    # Create train dataset and train dataloader
    train_datasets = MyDataset(meta_path, data_dir, data_transforms[mode])
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set model as evaluation mode
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        loss_sum = 0
        correct = 0

        for data in tqdm(train_dataloader):
            #x = transformed image, y = correct class
            x, y = data
            optimizer.zero_grad()
            # forward pass, calculate loss, update weight
            output = model(x.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
            # add loss
            _, preds = torch.max(output, 1)
            for j in range(32):
                if preds[j] == y[j]:
                    correct += 1
            loss_sum += loss.item()
            
        print("epoch[%d] loss: %.3f" %(epoch+1, loss_sum / 1250),
            "accuracy: {:.3f}".format(correct/40000 *100))
        torch.save(model.state_dict(), './drive/MyDrive/model.pth')


def get_model(model_name, checkpoint_path):
    
    model = model_name(res_block)
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    model = get_model(model_name, checkpoint_path)
    #test transform
    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #train transform
    transt = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(90, expand=False)])
    data_transforms = {
        'train' : transt, 
        'test' : trans
    }

    # Create test dataset, test dataloaders
    test_datasets = MyDatasett(meta_path, data_dir, data_transforms[mode])
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU or CPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Inference
    result = []
    for images, filename in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'filename': filename[i],
                'class': preds[i].item()
            })
    
    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    with open('./result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])


def main() :
    test()
    pass


if __name__ == '__main__':
    main()