from xml.parsers.expat import model
import torch
from data import dataset
from configure import config
from model import unet
from torch.utils.data import DataLoader

class train:
    def __init__(self):
        self.config = config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = unet.Unet().to(self.device)
        self.optmizer = torch.optim.Adam(params=self.model.parameters(),
        lr = self.config.learning_rate,
        betas = self.config.betas,
        eps = self.config.eps,
        weight_decay = self.config.weight_decay,
        amsgrad = self.config.amshrad)
        self.dataloader = DataLoader(dataset.train_data, self.config.batch_size)
        self.initialization()
        print(self.device)
    
    def dice_loss(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        intersection = pred * gt
        dice = (2 * intersection.sum(dim=(1,2)) ) / (pred.sum(dim=(1,2)) + gt.sum(dim=(1,2)))
        return 1-dice.mean()

    def train_step(self, x, gt):
        pred = self.model(x)
        loss = self.dice_loss(pred, gt)
        self.optmizer.zero_grad()
        loss.backward()
        self.optmizer.step()
        return loss

    def initialization(self):
        for module in (self.model).modules():    
            if (isinstance(module, ((torch.nn.Conv2d, torch.nn.ConvTranspose2d)))):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def train(self):
        print("---------Training start---------")
        for e in range(1, self.config.epochs + 1):
            accu_loss = 0
            for b,(x, gt) in enumerate(self.dataloader):
                loss = float(self.train_step(x, gt))
                accu_loss=accu_loss+loss
                print("-----------Ephochs: %d Batch: %d loss:"%(e,b),loss)
            # if (e % 5 == 0):
            torch.save(self.model.state_dict(), "weight.pth")

    
t = train()
t.train()