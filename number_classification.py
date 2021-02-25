 
import numpy as np
import torch 
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms 
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_set = datasets.MNIST("~/.pytorch/MNIST_data/",download=True,train=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
test_set = datasets.MNIST("~/.pytorch/MNIST_data",download=True,train=False,transform=transform)
testloader = torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=True)
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
train_losses, test_losses = [], []
epochs = 30
steps = 0 
for e in range(epochs):
    running_loss=0
    for images,labels in train_loader:
        optimizer.zero_grad()
        ps = model(images)
        loss = criterion(ps,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    test_loss = 0 
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            ps = model(images)
            test_loss += criterion(ps,labels)
            ps = torch.exp(ps)
            top_p, top_c = ps.topk(1,dim = 1)
            equals = top_c==labels.view(*top_c.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    model.train()
    train_losses.append(running_loss/len(train_loader))
    test_losses.append(test_loss/len(testloader))

    print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
