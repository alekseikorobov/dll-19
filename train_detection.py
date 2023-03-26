

# лос по центру, по площади, (по точкам на тесте - не реализовано)
# генерация изображений на тесте (несколько основных примеров)
# периодическое сохранение модели
# модель с наименьшими потерями сохранять отдельно и пересохранять при нахождении более лучшей


import os
import PIL
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.tensorboard import SummaryWriter
from MyDataset import MyDataset
from common.engine import train_one_epoch
from common import utils
import cv2
import shutil

from train_config_400_1 import configuration as conf_train
from test_config_400_1 import configuration as conf_test
from test_im_config_400_1 import configuration as conf_test_im
from utils.my_utils import get_array_loss, get_center_loss, get_count_box_loss
import MyModel as my_model
import torchvision
#from torchmetrics.detection.mean_ap import MeanAveragePrecision

to_image = torchvision.transforms.ToPILImage()

#params:
# our dataset has two classes only - background and text
num_classes = 2
num_epochs = 100
save_freq = 5
log_folder = 'output/logs/data_m_rcnn_1'
save_dir = 'output/model_1'
save_dir_best = 'output/model_best_1'
save_dir_testing_im = 'output/testing_image_1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 2


dataset = MyDataset(conf_train, is_train=True)
dataset_test = MyDataset(conf_test, is_train=False)
dataset_test_im = MyDataset(conf_test_im, is_train=False)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test_im = torch.utils.data.DataLoader(
    dataset_test_im, batch_size=batch_size, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

##################################################################################################
##################################################################################################



# get the model using our helper function
model = my_model.get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = None
    #torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
opt_name = 'adamw'
lr =0.005 #0.02
momentum=0.9
weight_decay=0.0005 #1e-4
if opt_name.startswith("sgd"):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov="nesterov" in opt_name,
    )
elif opt_name == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
# не сработало:
# optimizer = torch.optim.Adam(params, lr=0.005,
#                             weight_decay=0.0005)
# не сработало:
#optimizer = torch.optim.Adam(params, lr=0.05)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# let's train it for 10 epochs

if os.path.exists(log_folder):
    shutil.rmtree(log_folder)
if os.path.exists(log_folder):
    raise Exception(f'{log_folder=} already exists, please select another folder')

writer = SummaryWriter(log_folder)

def collback_loss(loss_value, step):
    writer.add_scalar('Loss/train', loss_value, step)

def convert_for_metrics(batch_size, predicts, targets):
    targets_new = []
    predicts_new = []
    for i in range(batch_size):
        predicts_new.append(
            dict(
                boxes=predicts[i]["boxes"],
                labels=predicts[i]["labels"],
                scores=predicts[i]["scores"],
            )
        )
    
    for i in range(batch_size):
        targets_new.append(
            dict(
                boxes=targets[i]["boxes"],
                labels=targets[i]["labels"],
            )
        )
    return predicts_new,targets_new

step = 0
def testing_model(path, data_loader_test):
    min_loss = np.inf
    #metrics = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
    global step
    print(f'testing_model {path=}')
    model = my_model.load_model(path, num_classes, device)
    model.eval()
    model.to(device)
    #losses_array = []
    #losses_box = []
    with torch.no_grad():    
        for images, targets in data_loader_test:
            step += 1
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predicts = model(images)
            
            boxes_predict = torch.cat(list(map(lambda m:m["boxes"], predicts)))
            boxes_fact = torch.cat(list(map(lambda m:m["boxes"],targets)))
            
            center_loss = get_center_loss(boxes_predict, boxes_fact)            
            center_loss = center_loss.item()
            array_loss = get_array_loss(boxes_predict, boxes_fact)
            array_loss = array_loss.item()                        
            count_box_loss = get_count_box_loss(torch.FloatTensor(list(map(lambda m:len(m["boxes"]),predicts))),
                torch.FloatTensor(list(map(lambda m:len(m["boxes"]),targets))))            
            count_box_loss = count_box_loss.item()
            
            pred, targ = convert_for_metrics(len(images),predicts,targets)            
            #metrics.update(pred, targ)
            #metr = metrics.compute()            
            #metr_map = metr['map'].item()                
            
            writer.add_scalar('Loss/test_array', array_loss, step)
            writer.add_scalar('Loss/test_center', center_loss, step)
            writer.add_scalar('Loss/test_box', count_box_loss, step)
            #writer.add_scalar('Loss/test_map', metr_map, step)
            
            min_loss = array_loss + center_loss + count_box_loss
            
    return min_loss
    #return np.array(losses_array).mean(), np.array(losses_box).mean()

iter_image = 0
def save_testing_images(model, save_dir_testing_im, data_loader_test_im, epoch):
    print(f'save_testing_images...')
    global iter_image
    model.eval()
    image_item = 0
    with torch.no_grad():
        for images, _ in data_loader_test_im:
            images = list(image.to(device) for image in images)            
            predicts = model(images)
            
            for img, pred in zip(images,predicts):                
                img_array = to_image(img.detach().cpu())
                for box in pred['boxes']:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    img_array = cv2.rectangle(np.asarray(img_array),(x1,y1),(x2,y2),color=(0,0,255))
                iter_image += 1
                image_item += 1
                im_path = os.path.join(save_dir_testing_im,f'im_{image_item}_{epoch}_{iter_image}.png')
                cv2.imwrite(im_path,cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                
            
            
            
    
min_global_loss = np.inf,np.inf
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=10, collback_loss=collback_loss)

    if epoch % save_freq == 0 and epoch > 0:
        path = my_model.save_model(model, epoch, lr_scheduler, optimizer, save_dir)
        my_model.remove_old_model(epoch, save_freq, save_dir)        
        min_loss = testing_model(path, data_loader_test)
        print(f'{min_loss=}')
        
        save_testing_images(model, save_dir_testing_im, data_loader_test_im, epoch)
                
        if min_loss < min_global_loss:
            my_model.save_model(model, epoch, lr_scheduler, optimizer, save_dir_best,f'MaskRCNN_resnet50_{opt_name}_min_loss.pth')
            min_global_loss = min_loss        

    # update the learning rate
    lr_scheduler.step()
    


writer.close()
