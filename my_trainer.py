from my_util import *
from my_net import *
import os
import datetime
import time
import csv
import copy

from torch import optim
import torch

import matplotlib.pyplot as plt

class My_trainer():
    def __init__(self, fpath, net, loss_func_name, batchsize, num_epoch, path_history, per_train=70, per_valid=15, per_test=15, y_label='RFS_MONTHS', gen_graph_func='UMAP', task='regression'):
        self.fpath = fpath
        self.per_train = per_train
        self.per_valid = per_valid
        self.per_test = per_test
        self.y_label = y_label
        self.g_func = gen_graph_func
        self.task = task
        self.df = pd.read_csv(fpath)
        self.split = Data_split(self.df, per_train, per_valid, per_test, y_label)
        if task=='regression':
            self.train_X, self.train_Y, self.valid_X, self.valid_Y, self.test_X, self.test_Y = self.split.split_for_regression()
        elif task=='classification':
            self.train_X, self.train_Y, self.valid_X, self.valid_Y, self.test_X, self.test_Y = self.split.split_for_classification()
        transform = Transform_data(self.train_X)
        self.edge_index = transform.generate_edge_index(gen_graph_func)
        list_train = generate_Data_List(x=self.train_X, edge_index=self.edge_index, y=self.train_Y)
        list_valid = generate_Data_List(x=self.valid_X, edge_index=self.edge_index, y=self.valid_Y)
        list_test = generate_Data_List(x=self.test_X, edge_index=self.edge_index, y=self.test_Y)
        self.batch_size = batchsize
        self.loader_train = generate_Data_Loader(list_train, batch_size=self.batch_size, shuffle=True)
        self.loader_valid = generate_Data_Loader(list_valid, batch_size=self.batch_size, shuffle=False)
        self.loader_test = generate_Data_Loader(list_test, batch_size=self.batch_size, shuffle=False)
        self.net = net
        self.loss_func_name = loss_func_name
        self.num_epoch = num_epoch
        self.optimizer = optim.Adam(self.net.parameters())
        self.best_acc = 0.0
        self.best_loss = 10000.0
        
        dt_now = datetime.datetime.now()
        dir_name = dt_now.strftime('%Y_%m_%d_%H%M_datalog')
        self.path_hist_dir = path_history+'/'+dir_name
        os.mkdir(self.path_hist_dir)
        self.f_param = open(self.path_hist_dir+'/parameters.txt', 'w')
        self.f_param.write(dir_name+'\n')
        self.f_param.write(f'fpath: {self.fpath}\n')
        self.f_param.write(f'Y_label: {self.y_label}\n')
        self.f_param.write(f'per_train: {self.per_train}%, per_valid: {self.per_valid}%, per_test: {self.per_test}%\n')
        self.f_param.write('\n')
        self.f_param.write(f'task: {self.task}\n')
        self.f_param.write(f'graph genertion func: {self.g_func}\n')
        self.f_param.write('\n')
        self.f_param.write(f'loss function name: {self.loss_func_name}\n')
        self.f_param.write(f'batchsize: {self.batch_size}\n')
        self.f_param.write(f'epoch: {self.num_epoch}\n')
        self.f_param.close()
        
    def train(self, net, dataloader, optimizer):
        net.train()
        total_loss = 0.0
        total_correct = 0
        accuracy = 0.0
        pred_list = []#
        
        for data in dataloader:
            data.to(set_device())
            optimizer.zero_grad()
            out = net(data)
            loss = loss_fnc(out, data.y, self.loss_func_name)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if self.task=='classification':
                pred = out.argmax(dim=1)
                total_correct += (pred==data.y.argmax(dim=1)).sum().item()
                pred_list+=torch.flatten(pred).tolist()#
            elif self.task=='regression':
                pred_list+=torch.flatten(out).tolist()
        avg_loss = total_loss/len(dataloader.dataset)
        print(pred_list)
        if self.task=='classification':
            accuracy = total_correct/len(dataloader.dataset)
            if accuracy>self.best_acc:
                self.best_acc = accuracy
                torch.save(net.state_dict(), self.path_hist_dir+'/BestModel.pth')
                self.best_net = copy.deepcopy(net)
                print('This model is saved.')
        elif self.task=='regression':
            if avg_loss<self.best_loss:
                self.best_loss = avg_loss
                torch.save(net.state_dict(), self.path_hist_dir+'/BestModel.pth')
                self.best_net = copy.deepcopy(net)
                print('This model is saved.')
        return avg_loss, accuracy
    
    def validate(self, net, dataloader):
        accuracy = 0.0
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            for data in dataloader:
                data.to(set_device())
                out = net(data)
                loss = loss_fnc(out, data.y, self.loss_func_name)
                total_loss += loss.item()
                if self.task=='classification':
                    pred = out.argmax(dim=1)
                    total_correct += (pred==data.y.argmax(dim=1)).sum().item()
        avg_loss = total_loss/len(dataloader.dataset)
        if self.task=='classification':
            accuracy = total_correct/len(dataloader.dataset)
        return avg_loss, accuracy
    
    def test(self, net2, net_path, dataloader):
        # net = net2
        # net.load_state_dict(torch.load(net_path))
        net = self.best_net
        accuracy = 0.0
        net.eval()
        pred_list = []
        truth_list = []
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            for data in dataloader:
                data.to(set_device())
                out = net(data)
                loss = loss_fnc(out, data.y, self.loss_func_name)
                total_loss += loss.item()
                if self.task=='classification':
                    pred = out.argmax(dim=1)
                    total_correct += (pred==data.y.argmax(dim=1)).sum().item()
                    pred_list+=torch.flatten(pred).tolist()
                    truth_list+=torch.flatten(data.y.argmax(dim=1)).tolist()
                else:
                    pred_list+=torch.flatten(out).tolist()
                    truth_list+=torch.flatten(data.y).tolist()
        avg_loss = total_loss/len(dataloader.dataset)
        if self.task=='classification':
            accuracy = total_correct/len(dataloader.dataset)
        print('test done.')
        print(pred_list)
        print(truth_list)
        f = open(self.path_hist_dir+'/pred.csv','w')
        writer = csv.writer(f)
        writer.writerow(pred_list)
        f.close()
        f = open(self.path_hist_dir+'/truth.csv','w')
        writer = csv.writer(f)
        writer.writerow(truth_list)
        f.close()
        return avg_loss, accuracy
    
    def do_train_and_validate(self):
        print('Start training')
        start = time.time()
        history = {}
        history['train_loss_values'] = []
        history['train_accuracy_values'] = []
        history['valid_loss_values'] = []
        history['valid_accuracy_values'] = []
        
        for epoch in range(1, self.num_epoch+1):
            print(f'epoch: {epoch:2}')
            t_loss, t_accu = self.train(self.net, self.loader_train, self.optimizer)
            v_loss, v_accu = self.validate(self.net, self.loader_valid)
            print(f'train_loss: {t_loss:.6f}, train_accuracy: {t_accu:3.4%},', f'valid_loss: {v_loss:.6f}, valid_accuracy: {v_accu:3.4%}')
            history['train_loss_values'].append(t_loss)
            history['valid_loss_values'].append(v_loss)
            if self.task=='classification':
                history['train_accuracy_values'].append(t_accu)
                history['valid_accuracy_values'].append(v_accu)
        end = time.time()
        print(f'Finished training, time: {end-start}')
        return history

    def do_train_for_test(self, net2):
        print('Start training')
        start = time.time()
        history = {}
        history['train_loss_values'] = []
        history['train_accuracy_values'] = []
        history['test_loss_values'] = []
        history['test_accuracy_values'] = []
        # best_acc = 0.0

        for epoch in range(1, self.num_epoch+1):
            print(f'epoch: {epoch:2}')
            t_loss, t_accu = self.train(self.net, self.loader_train, self.optimizer)
            print(f'train_loss: {t_loss:.6f}, train_accuracy: {t_accu:3.4%}.')
            history['train_loss_values'].append(t_loss)
            if self.task=='classification':
                history['train_accuracy_values'].append(t_accu)
            # if best_acc<t_accu:
                # torch.save(self.net.state_dict(), self.path_hist_dir+'/BestModel.pth')
        end = time.time()
        print(f'Finished training, time: {end-start}')
        v_loss, v_accu = self.test(net2, self.path_hist_dir+'/BestModel.pth', self.loader_test)
        history['test_loss_values'].append(v_loss)
        if self.task=='classification':
            history['test_accuracy_values'].append(v_accu)
        print('Saving is finished.')
        return history
    
    def plot_graph(self, values1, values2, mg, label1, label2, save=False, graph_name=None):
        plt.plot(range(mg), values1, label=label1)
        plt.plot(range(mg), values2, label=label2)
        plt.legend()
        plt.grid()
        if save==True:
            plt.savefig(self.path_hist_dir+'/'+graph_name+'.png')
        plt.show()

    def save_loss_log(self, test_loss_list, valid_loss_list, num):
        df_test = pd.DataFrame(test_loss_list).T
        df_valid = pd.DataFrame(valid_loss_list).T
        name_test = self.path_hist_dir+'/'+str(num)+'_test_loss_log.csv'
        name_valid = self.path_hist_dir+'/'+str(num)+'_valid_loss_log.csv'
        df_test.to_csv(name_test)
        df_valid.to_csv(name_valid)

    def save_acc_log(self, test_acc_list, valid_acc_list, num):
        df_test = pd.DataFrame(test_acc_list).T
        df_valid = pd.DataFrame(valid_acc_list).T
        name_test = self.path_hist_dir+'/'+str(num)+'_test_acc_log.csv'
        name_valid = self.path_hist_dir+'/'+str(num)+'_valid_acc_log.csv'
        df_test.to_csv(name_test)
        df_valid.to_csv(name_valid)