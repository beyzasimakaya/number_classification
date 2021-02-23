 
import numpy as np
import torch 
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms 
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_set = datasets.MNIST("~/.pytorch/MNIST_data/",download=True,train=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,256)
        self.layer2 = nn.Linear(256,128)
        self.layer3 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64,10)
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.log_softmax(self.layer4(x),dim=1)
        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.003)
epochs = 5 
for e in range(epochs):
    running_loss=0
    correct = 0
    for images,labels in train_loader:
        ps = model(images)
        loss = criterion(ps,labels)
        optimizer.zero_grad()
        _,pred_label = torch.max(ps, dim = 1)
        correct = (pred_label == labels).float()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    
    accuracy = correct.sum() / len(correct)
    print(f"Accuracy = {accuracy}")
    print(f"Training Loss : {running_loss/len(train_loader)}")
    