import os,sys
import PIL
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.tensorboard import SummaryWriter
from MyDatasetRec import MyDatasetRec,MyTruncateTransform
from common import utils
import cv2
import MyModelRec as my_model
import torchvision
import config_rec
from test_config_400_1_rec import configuration
from utils.my_utils import get_text_coss_loss
from traceback import format_exc
from itertools import groupby
from tqdm import tqdm
import shutil
import torch.nn as nn
from torch.optim import lr_scheduler
import torchmetrics

class FitRecognizer:
    def __init__(self):
        self.dataset = MyDatasetRec(config_rec.train_conf, config_rec.all_alph)
        print(f'{len(self.dataset)=}')
        # for _ in dataset:
        #     pass

        train_size = int(len(self.dataset) * 0.7)
        test_size = len(self.dataset) - train_size
        print(f'{train_size=}, {test_size=}')

        train_set, val_set = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        #dataset_test = MyDatasetRec(config_rec.conf,all_alph, is_train=False)

        self.batch_size = 200
        # define training and validation data loaders
        self.data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.data_loader_test = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)   
        
        
        #path = 'output/model_rec_/CRNN_LSTM_v1_140.pth'
        
        
    
    
        #to_image = torchvision.transforms.ToPILImage()

        #params:
        # our dataset has two classes only - background and text
        self.num_epochs = 2000
        self.save_freq = 5
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'{self.device=}')
        self.blank_label = 0
        self.image_height = 28
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.cnn_output_height = 4
        self.cnn_output_width = config_rec.max_length #количество максимальной длины, которую сеть может предсказать, зависит от разрмера картики
        self.digits_per_sequence = 6
        self.number_of_sequences = 10
        self.num_classes = len(config_rec.all_alph) + 1

        # get the model using our helper function
        #model = my_model.CRNN(cnn_output_height, gru_hidden_size,gru_num_layers,num_classes)

        self.model = my_model.CRNN_v1(imgH=32,in_channels=3, nclass=self.num_classes, gru_size=256)

        # move model to the right device
        self.model.to(self.device)

        self.criterion = None
        if self.model.version == 'v0':
            self.criterion = nn.CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        elif self.model.version == 'v1':
            self.criterion = nn.CTCLoss(blank=self.blank_label, reduction='sum', zero_infinity=True)

        assert self.criterion, 'criterion must be init'
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)

        self.save_dir = 'output/model_rec_scale_2'
        self.save_dir_best = 'output/model_rec_text_best_scale_2'
        self.log_folder = f'output/logs/model_rec/data_crnn_LSTM_{self.gru_num_layers}_{self.model.version}_scale_8_24_2'

        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)


        if os.path.exists(self.log_folder):
            raise Exception(f'{self.log_folder=} already exists, please select another folder')

        self.writer = SummaryWriter(self.log_folder)

        self.validation_step = 0
        self.train_iter = 0
        self.validate_iter = 0
        #self.my_transforms = MyTruncateTransform()
    
    def fit(self):
        self.train(self.model, self.num_epochs, self.scheduler, self.data_loader, self.data_loader_test)
    
        
    def train(self, model, epochs, scheduler, train_loader, validation_loader, path = None):
        
        if path is None:
            print(f'start training')
        else:
            model = my_model.load_model(model,path,self.device)
            model.train()
            model.to(self.device)
            print(f'continue training')
        #global train_iter, validate_iter
        min_loss = np.inf
        for epoch in range(epochs):
            #scheduler.step()
            test_correct, test_total = 0, 0

            for x_train, y_train in tqdm(train_loader, position=0, leave=True, file=sys.stdout):
                self.train_iter += 1
                #print(f'{x_train.shape, y_train.shape=}')
                batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
                #print(f'{batch_size=}')
                #x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
                self.optimizer.zero_grad()
                y_pred = model(x_train.to(self.device))
                #print(f'{y_pred1.shape=}') #y_pred.shape=torch.Size([3, 5, 12]) N, T, C: N - Batch size, T - Input sequence length, C - Number of classes (including blank)
                ##print(f'{y_pred1=}')
                #y_train = y_train[y_train>0]
                loss = None
                if model.version == 'v0':
                    y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([5, 3, 12]) T, N, C
                    input_lengths = torch.IntTensor(batch_size).fill_(self.cnn_output_width)
                    target_lengths = torch.IntTensor([len([t1 for t1 in t if t1 != 0] ) for t in y_train])
                    #print(f'{input_lengths=}')
                    #print(f'{target_lengths=}')
                    loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
                elif model.version == 'v1':
                    preds_size = torch.IntTensor([y_pred.size(0)] * batch_size)  # seqLength x batchSize
                    #print(f'{preds_size=}')
                    target_lengths = torch.IntTensor([len([t1 for t1 in t if t1 != 0] ) for t in y_train])        
                    loss = self.criterion(y_pred.log_softmax(2).cpu(), y_train, preds_size, target_lengths) / batch_size
                
                if self.writer is not None:
                    
                    self.writer.add_scalar('train/loss',loss.item(), self.train_iter)

                loss.backward()
                self.optimizer.step()

                t_c, t_t = self.testing(y_train, batch_size, y_pred)
                
                test_correct += t_c
                test_total += t_t

            test_acc = test_correct / test_total
            print(f'TRAINING {epoch=} {scheduler.get_lr()=}. Correct: {test_correct=} / {test_total=} {test_acc:.3f}')

            self.writer.add_scalar('test/accuracy',test_acc, epoch)
            
            if epoch % self.save_freq == 0 and epoch > 0:
                self.validate_iter += 1
                path = my_model.save_model(model, epoch, scheduler, self.optimizer, self.save_dir)

                my_model.remove_old_model(model, epoch, save_freq=self.save_freq, save_dir=self.save_dir)

                validation_metrics = self.validation_model(model.version, path, validation_loader,model.num_classes,self.dataset.torch_text_dict, self.writer)
                
                validation_similarly_loss_list_mean = validation_metrics['similarly_loss_list_mean']
                validation_similarly_loss_list_std = validation_metrics['similarly_loss_list_std']
                validation_accuracy = validation_metrics['accuracy']
                
                validation_min = validation_similarly_loss_list_mean + validation_similarly_loss_list_std + (1 - validation_accuracy)            
                
                self.writer.add_scalar('validate/similarly_loss_mean', validation_similarly_loss_list_mean, self.validate_iter)
                self.writer.add_scalar('validate/similarly_loss_std', validation_similarly_loss_list_std, self.validate_iter)
                self.writer.add_scalar('validate/accuracy', validation_accuracy, self.validate_iter)
                
                if validation_min < min_loss:
                    min_loss = validation_min
                    my_model.save_model(model, epoch, scheduler, self.optimizer, self.save_dir_best, f'CRNN_{model.gru_name}_min_loss.pth')
                if validation_min == 0 and test_correct == test_total:
                    print('finish train!')
                    return

    def testing(self, y_train, batch_size, y_pred):
        train_correct, train_total = 0, 0
        _, max_index = torch.max(y_pred, dim=2)  # max_index.shape == torch.Size([32, 64])
                ##print(f'{max_index.shape=}')
        for i in range(batch_size):
                    #y_train_i = torch.Tensor([t1 for t1 in y_train[i] if t1 != 0])
            y_train_i = torch.IntTensor([c for c in y_train[i] if c != 0])
                    
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32            
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != self.blank_label])
            #txt = dataset.torch_text_dict.get_label(prediction)
            #print(f'{txt=}')
            # print(f'{y_train_i=}')
            # print(f'{prediction=}')
            if len(prediction) == len(y_train_i):
                if torch.all(prediction.eq(y_train_i)):
                    train_correct += 1
            train_total += 1
        return train_correct, train_total


    
    #cer = torchmetrics.CharErrorRate()
    def validation_model(self, version, path, data_loader_test, num_classes,torch_text_dict, writer = None):
        #global validation_step
        
        validation_correct, validation_total = 0, 0
        
        print(f'testing_model {path=}')
        model = None
        if version == 'v0':
            #model = my_model.CRNN(cnn_output_height, gru_hidden_size,gru_num_layers,num_classes)
            ...
        elif version == 'v1':
            model = my_model.CRNN_v1(imgH=32,in_channels=3, nclass=num_classes, gru_size=256).to(self.device)

        model = my_model.load_model(model, path, self.device)
        model.eval()
        model.to(self.device)
        similarly_loss_list = []
        char_error_rate_list = []
        min_similarly_loss = np.inf
        with torch.no_grad():    
            for x_train, y_train in data_loader_test:
                batch_size = x_train.shape[0]
                y_pred = model(x_train.to(self.device) )
                if model.version == 'v0':
                    y_pred = y_pred.permute(1, 0, 2)
                _, max_index = torch.max(y_pred, dim=2)

                for i in range(batch_size):
                    self.validation_step +=1
                    #y_train_i = torch.Tensor([t1 for t1 in y_train[i] if t1 != 0])
                    
                    y_train_i = torch.IntTensor([c for c in y_train[i] if c != 0])
                    #print(f'{y_train_i.shape}')
                    raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32            
                    
                    prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != self.blank_label])
                    #print(f'{prediction.shape}')
                    if len(prediction) == len(y_train_i):
                        if torch.all(prediction.eq(y_train_i)):
                            validation_correct += 1
                    validation_total += 1

                    similarly_loss = get_text_coss_loss(prediction, y_train_i)
                    similarly_loss = similarly_loss.item()
                    
                    #char_error_rate = cer()
                    #char_error_rate_list = char_error_rate
                    
                    if writer is not None:
                        writer.add_scalar('test/similarly_loss', similarly_loss, self.validation_step)
                    if similarly_loss < min_similarly_loss:
                        min_similarly_loss = similarly_loss
                        #print(f'{prediction=}')
                        txt_predict = torch_text_dict.get_label(prediction[0])
                        txt_fact = torch_text_dict.get_label(y_train_i[0])
                        print(f'{txt_predict=}, {txt_fact=}')
                        
                    similarly_loss_list.append(similarly_loss)
                    
            validation_acc = validation_correct / validation_total
            print(f'{validation_correct} / {validation_total} = {validation_acc:.3f}')

        return {
            'similarly_loss_list_mean':np.array(similarly_loss_list).mean(),
            'similarly_loss_list_std':np.array(similarly_loss_list).std(),
            'accuracy': validation_acc        
        }


if __name__ == '__main__':    
    fit_rec = FitRecognizer()    
    fit_rec.fit()