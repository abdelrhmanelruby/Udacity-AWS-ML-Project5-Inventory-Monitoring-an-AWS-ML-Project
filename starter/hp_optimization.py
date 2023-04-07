import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os
from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd
import logging
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import sys
logger.addHandler(logging.StreamHandler(sys.stdout))
from tqdm import tqdm 


def test(model, test_loader, criterion):
    logger.debug(f"Testing model on whole testing dataset")
    model.eval()
    loss=0
    corrects=0
    
    for i, X in tqdm(test_loader):
        outputs=model(i)
        loss=criterion(outputs, X)
        _, preds = torch.max(outputs, 1)
        loss += loss.item() * i.size(0)
        corrects += torch.sum(preds == X.data)

    t_loss = loss / len(test_loader.dataset)
    t_acc = corrects.double() / len(test_loader.dataset)
    
    logger.info(f"Testing Loss: {t_loss}")
    logger.info(f"Testing Accuracy: {t_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer):
    epochs=1
    best_loss=1e6
    dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for step in ['train', 'valid']: 
            logger.info(f"Epoch: {epoch}, step {step}")
            if step=='train':
                model.train()
            else:
                model.eval()
            loss = 0.0
            corrects = 0.0
            samples=0
            
            t_sample_in_this_step = len(dataset[step].dataset)

            for i, X in dataset[step]:
                outputs = model(i)
                loss = criterion(outputs, X)

                if step=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                loss += loss.item() * i.size(0)
                corrects += torch.sum(preds == X.data)
                samples+=len(i)
                
                accuracy = corrects/samples
                logger.debug("Epoch {}, Phase {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        epoch,
                        step,
                        samples,
                        t_sample_in_this_step,
                        100.0 * (float(samples) / float(t_sample_in_this_step)),
                        loss.item(),
                        corrects,
                        samples,
                        100.0*accuracy,
                    ))
                

                if (samples>(0.1*t_sample_in_this_step)):
                    break                  

            epoch_loss = loss / samples
            epoch_acc = corrects / samples
            
            if step=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1


            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(step,
                                                                                 epoch_loss,
                                                                                 epoch_acc,
                                                                                 best_loss))          
            
        if loss_counter==1:
            logger.info(f"Break : increasing in loss at epoch: {epoch}")
            break
        if epoch==0:
            logger.info(f"Break : limit at epoch: {epoch}")
            break
    return model
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 5))
    return model

def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    training_data = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    testing_data = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    testing_data_loader  = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return training_data_loader, testing_data_loader, validation_data_loader

def main(args):
    logger.info(f'HP are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    model=net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    logger.info("Starting model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer)
    
    logger.info("Testing model")
    test(model, test_loader, criterion)
    
    logger.info("Saving model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Hyperparameter optimization script",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    logger.debug(f'Arguments: {args}')
    
    main(args)
