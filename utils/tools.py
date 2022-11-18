# Original code from https://github.com/zhouhaoyi/Informer2020/
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from json.decoder import JSONDecodeError
from json import JSONEncoder


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.episode_count = 0

    def __call__(self, season, episode, val_mse, val_mae, preds, trues, model, path):
        score = -val_mae
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_mae, model, path)
            self.save_data(season, episode, val_mse, val_mae, preds, trues)
        elif score < self.best_score + self.delta:
            if self.episode_count == 0:
              self.save_checkpoint(val_mae, model, path)
              self.save_data(season, episode, val_mse, val_mae, preds, trues)
            self.counter += 1
            self.episode_count +=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_mae, model, path)
            self.save_data(season, episode, val_mse, val_mae, preds, trues)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/checkpoints/'+'checkpoint.pth')
        self.val_loss_min = val_loss

    
    def save_data(self, season, episode, val_mse, val_mae, preds, trues):

      if season == 1 and episode == 1:
        with open('datasave.json','w+') as file1: 
          file_data = {}
          file_data["s" + str(season) + "e" + str(episode)] = {}
          file_data["s" + str(season) + "e" + str(episode)]["val_mse"] = float(val_mse)
          file_data["s" + str(season) + "e" + str(episode)]["val_mae"] = float(val_mae)
          file_data["s" + str(season) + "e" + str(episode)]["preds"] = preds
          file_data["s" + str(season) + "e" + str(episode)]["trues"] = trues
          json.dump(file_data, file1, indent=4, cls=NumpyArrayEncoder)
          file1.close()
      else:
        filename = 'datasave.json'
        with open(filename, 'r') as f:
            file_data = json.load(f)
            file_data["s" + str(season) + "e" + str(episode)] = {}
            file_data["s" + str(season) + "e" + str(episode)]["val_mse"] = float(val_mse)
            file_data["s" + str(season) + "e" + str(episode)]["val_mae"] = float(val_mae)
            file_data["s" + str(season) + "e" + str(episode)]["preds"] = preds
            file_data["s" + str(season) + "e" + str(episode)]["trues"] = trues

        os.remove(filename)
        with open(filename, 'w') as f:
            json.dump(file_data, f, indent=4, cls=NumpyArrayEncoder)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        #Save mean and std for off-the-cuff inversal of transform
        np.save('settings/std.npy',std )
        np.save('settings/mean.npy',mean)
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe


def visualize_predictions(v_preds, v_trues):
    #Validation content
    print("Feature count below")
    print(len(v_preds[0][0][0])) #7 features
    print("Timestep count below")
    print(len(v_preds[0][0])) #24 timesteps
    print("Batch_size below")
    print(len(v_preds[0])) #32 batch_size
    print("Number of predictions")
    print(len(v_preds)) #2842 validation steps

    #Get prediction values
    featurepred1 = []
    featurepred2 = []
    featurepred3 = []

    multiOutput = False

    if len(v_preds[0][0][0]) > 1 or len(v_trues[0][0][0]) > 1:
        multiOutput = True
    else: 
        multiOutput = False

    #Load Std and mean to inverse transformation of scaling just for prediction
    s = np.load('settings/std.npy')
    m = np.load('settings/mean.npy')

    for feature in v_preds[0][0]:
        featurepred1.append((feature[0]*s[0])+m[0])
        if multiOutput:
            featurepred2.append((feature[1]*s[1])+m[1])
            featurepred3.append((feature[2]*s[2])+m[2])

    #Get truth values
    featuretrue1 = []
    featuretrue2 = []
    featuretrue3 = []

    for feature in v_trues[0][0]:
        featuretrue1.append((feature[0]*s[0])+m[0])
        if multiOutput:
            featuretrue2.append((feature[1]*s[1])+m[1])
            featuretrue3.append((feature[2]*s[2])+m[2])

    #Map features to variables and graphs
    y = featurepred1 
    z = featuretrue1

    if multiOutput:
        a = featurepred2 
        b = featuretrue2
        c = featurepred3 
        d = featuretrue3

    plt.plot(y, label='Prediction')
    plt.plot(z, label='Truth')

    plt.title('NP15')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('images/NP15.png')
    plt.close()

    if multiOutput:
        plt.plot(a, label='Prediction')
        plt.plot(b, label='Truth')

        plt.title('SP15')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('images/SP15.png')
        plt.close()

        plt.plot(c, label='Prediction')
        plt.plot(d, label='Truth')

        plt.title('ZP26')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('images/ZP26.png')
        plt.close()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        axes[0].plot(y)
        axes[0].plot(z)
        axes[1].plot(a)
        axes[1].plot(b)
        axes[2].plot(c)
        axes[2].plot(d)
        fig.tight_layout()
        plt.savefig('images/predictions.png')
        plt.close()


def visualize_loss(train_mses, train_maes, val_mses, val_maes):
    plt.plot(train_mses, label='train_mse')
    plt.plot(train_maes, label='train_mae')

    plt.title('training mse and mae')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/train-mse-and-mae.png')
    plt.close()

    plt.plot(val_mses, label='val_mse')
    plt.plot(val_maes, label='val_mae')

    plt.title('validation mse and mae')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/val-mse-and-mae.png')
    plt.close()