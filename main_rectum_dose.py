
# coding: utf-8

# In[ ]:


import os, sys, pdb, random, math, datetime, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm

from dataloaders.dose_loader_rectum import Dose_online_avoid
import dataloaders.dose_transforms as tr
import matplotlib.pyplot as plt

from networks.dose.deeplab_resnet import DeepLabv3_plus_resnet
from networks_cfgs.deeplab import deeplab_res50_cfg
from utils import get_logger
import tasks.rectum_tasks as rectum_tasks

from losses import VGG_PerceptualLoss
# In[ ]:


class Config(object):
    def __init__(self):
        
        self.train_batch = 20
        self.validation_batch = 10
        self.dataset = "rectum"
        self.data_root = "./data/rectum"
        self.train_txt = "./datalist/temp.txt"
        self.validation_txt = "./datalist/temp.txt"
        
        self.nepoch = 100
        self.HU_max = 390 
        self.HU_min = -310 
        self.prescription = 50.4
        
        self.mask_dict = rectum_tasks.mask_dict_5OARS #mask_dict_4OARS | mask_dict_all
        
        self.s_h = 196 # target height
        self.s_w = 196
        self.rotation = 15
        
        self.optimizer = "sgd"
        self.lr = 0.01# 1.0*1e-7
        self.in_channels = len(self.mask_dict) + 1 #3 | len(self.mask_dict)
        self.num_classes = 1
        self.p_losses = ["VGG_PerceptionLoss"] #"VGG_PerceptionLoss"
        self.wd = 5e-4
        self.momentum = 0.9
         
        self.network = 'DeepLabv3_plus_resnet'
        self.net_config = deeplab_res50_cfg #res_unet50_regularize_cfg([0.5, 0.5, 0.5])
        
        self.criterion = "mse" # cross_entropy | dice
        
        self.suffix = "vanilla" #inception_perception_sgd_aug
        self.checkpoint = None
        
        self.gpus = "0"
        self.num_workers = 4
        
        self.manualSeed = None
        
config = Config()


# In[ ]:


log_path = os.path.join('logs', config.dataset,  config.network, '{}.log'.format(config.suffix))
if os.path.exists(log_path):
    delete_log = input("The log file %s exist, delete it or not (y/n) \n"%(log_path))
    if delete_log in ['y', 'Y']:
        os.remove(log_path)
    else:
        log_path = os.path.join('logs', config.dataset, config.network, '{}_{}.log'.format(config.suffix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

checkpoint_path = os.path.join('checkpoint', config.dataset, config.network, config.suffix)
if os.path.exists(checkpoint_path):
    delete_checkpoint_path = input("The checkpoint folder %s exist, delete it or not (y/n) \n"%(checkpoint_path))
    if delete_checkpoint_path in ['y', 'Y']:
        shutil.rmtree(checkpoint_path)
    else:
        checkpoint_path = os.path.join("checkpoint", config.dataset, config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    os.makedirs(checkpoint_path)

summary_path = os.path.join("summaries", config.dataset, config.network, config.suffix)
if os.path.exists(summary_path):
    delete_summary = input("The tf_summary folder %s exist, delete it or not (y/n) \n"%(summary_path))
    if delete_summary in ['y', 'Y']:
        shutil.rmtree(summary_path)
    else:
        summary_path = os.path.join("summaries", config.dataset, config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    os.makedirs(summary_path)
    
logger = get_logger(log_path)
writer = SummaryWriter(summary_path)
logger.info(config.__dict__)

if config.manualSeed is None:
    config.manualSeed = random.randint(1, 10000)
logger.info("Random Seed: {}".format(config.manualSeed))
np.random.seed(config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)


# In[ ]:


def log_best_metric(metric_list, cur_epoch_idx, logger, state, save_path, save_model=True, metric = "mse"):
    if len(metric_list) == 0:
        return
    else:
        best_idx = np.argmin(metric_list)
        best_metric = metric_list[best_idx]
        if best_idx == cur_epoch_idx:
            logger.info("Epoch: %d, Validation %s improved to %.8f"%(cur_epoch_idx, metric, best_metric))
            if save_model:
                dir_path = os.path.dirname(save_path)  # get parent path
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(state, save_path)
                logger.info("Model saved in file: %s"%(save_path))
        else:
            logger.info("Epoch: %d, Validation %s didn't improve. Best is %.8f in epoch %d"%(cur_epoch_idx, metric, best_metric, best_idx))


# In[ ]:


def train(model, device, data_loader, criterion, optimizer, p_criterions, epoch, writer):
    model.train()
    losses = []
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, sample_batched in enumerate(data_loader):
            inputs, labels = sample_batched['input'], sample_batched['rd_slice']
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 1)
            mse_loss = criterion(outputs, labels)
            
            p_losses = 0
            for i, p_criterion in enumerate(p_criterions):
                p_losses += p_criterion(torch.unsqueeze(outputs, 1), torch.unsqueeze(labels, 1))
                
            loss = mse_loss + p_losses
            
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            pbar.set_description("Epoch %d, Batch %d/%d, Train loss: %.4f"%(epoch, batch_idx+1, len(data_loader), np.mean(losses)))
            
            inputs = inputs.cpu().numpy()
    
    ave_loss = np.mean(losses)
    writer.add_scalar('train/epoch_loss', ave_loss, epoch)
    return ave_loss

def validate(model, device, data_loader, criterion, epoch, writer):
    losses = []
    model.eval()
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(tqdm(data_loader)):
            inputs, labels = sample_batched['input'], sample_batched['rd_slice']
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
                
    ave_loss = np.mean(losses)
    writer.add_scalar('validation/epoch_loss', ave_loss, epoch)
                
    return ave_loss


# In[ ]:


train_tr = transforms.Compose([
            tr.FilterHU(config.HU_min, config.HU_max),
            tr.NormalizeCT(config.HU_min, config.HU_max),
            tr.Arr2image(),
            tr.AlignCT(),
            tr.Padding([config.s_h, config.s_w]), # h, w
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(config.rotation),
            tr.NormalizeDosePerSample(),
            tr.Stack2Tensor()
        ])

validation_tr = transforms.Compose([
            tr.FilterHU(config.HU_min, config.HU_max),
            tr.NormalizeCT(config.HU_min, config.HU_max),
            tr.Arr2image(),
            tr.AlignCT(),
            tr.Padding([config.s_h, config.s_w]), # h, w
            tr.NormalizeDosePerSample(),
            tr.Stack2Tensor()
        ])

trainset = Dose_online(config.data_root, 
                        sample_txt = config.train_txt,
                        mask_dict = config.mask_dict,
                        transforms = train_tr)

validationset = Dose_online(config.data_root, 
                        sample_txt = config.validation_txt,
                        mask_dict = config.mask_dict,
                        transforms = validation_tr)

trainset_loader = DataLoader(trainset, batch_size=config.train_batch, shuffle=True, num_workers=config.num_workers)
validationset_loader = DataLoader(validationset, batch_size=config.validation_batch, shuffle=False, num_workers=config.num_workers)
logger.info("Number of training images: {}, validation images: {}".format(len(trainset), len(validationset)))

# In[ ]:


model = globals()[config.network](config.in_channels, config.num_classes, config.net_config)

lr_scheduler = None
if config.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.wd)
    lr_lambda = lambda epoch: (1 - float(epoch) / config.nepoch)** 0.9
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif config.optimizer == "adadelta":
    optimizer = optim.Adadelta(model.parameters(), lr = config.lr)
else:
    raise("Unknown optimizer: {}".format(config.optimizer))
    
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
gpus = range(len(config.gpus.split(",")))
if len(gpus) > 1:
    model = nn.DataParallel(model, gpus)
device = torch.device("cuda:{}".format(gpus[0]))
model.to(device)

if config.criterion == "mse":
    criterion = nn.MSELoss(size_average = True, reduce = True)
    p_criterions = [globals()[loss]() for loss in config.p_losses]
else:
    raise("Unknown criterion: {}".format(config.criterion))


# In[ ]:

metric_list = []
for epoch in range(config.nepoch):
    if lr_scheduler is not None:
        lr_scheduler.step()
        logger.info("Epoch: %d, Learning rate: %.10f"%(epoch, lr_scheduler.get_lr()[0]))
    train_loss = train(model, device, trainset_loader, criterion, optimizer, p_criterions, config.p_ratios, epoch, writer)
    logger.info("Epoch: %d, Train Loss: %.4f"%(epoch, train_loss))
    
    validation_loss = validate(model, device, validationset_loader, criterion, epoch, writer)
    metric_list.append(validation_loss)
    logger.info("Epoch: %d, Val Loss: %.8f"%(epoch, validation_loss))
    
    log_best_metric(metric_list, epoch, logger, 
                    {'epoch': epoch,
                     'state_dict': model.state_dict()},
                     '{}/epoch{}.pth'.format(checkpoint_path, epoch),
                    save_model=True,
                    metric = "mse")