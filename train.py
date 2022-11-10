
import time

import numpy as np
import torch


from torch.optim import Adam
from torch.utils.data import DataLoader

import deepspeed
import torch.nn as nn

import ipc
from config import build_parser
from model import Transformer
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from metrics import metric

import matplotlib.pyplot as plt

from utils.tools import EarlyStopping


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, enc_attn_type=args.encoder_attention,
                       dec_attn_type=args.decoder_attention, dropout=args.dropout)


def get_params(mdl):
    return mdl.parameters()


def _get_data(args, flag):
    if not args.data == 'ETTm1':
        Data = Dataset_ETT_hour
    else:
        Data = Dataset_ETT_minute
    # timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False;
        drop_last = True;
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
        # freq = args.detail_freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        # freq = args.freq

    data_set = Data(
        root_path='data',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        # timeenc=timeenc,
        # freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def run_metrics(caption, preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}'.format(caption, mse, mae))
    return mse, mae


def run_iteration(model, loader, args, training=True, message = ''):
    preds = []
    trues = []
    total_loss = 0
    elem_num = 0
    steps = 0
    target_device = 'cuda:{}'.format(args.local_rank)
    
    #if training == False:
    #  early_stopping = EarlyStopping(verbose=True) #7 as default <- Validation Early-stop
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if not args.deepspeed:
            model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32,
                              device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)

        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')

        #pred = result.detach().cpu().unsqueeze(2).numpy()  # .squeeze()
        pred = result.detach().cpu().numpy()  # .squeeze()
        true = target.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)
        trues.append(true)

        unscaled_loss = loss.item()
        total_loss += unscaled_loss
        print("{} Loss at step {}: {}, mean for epoch: {}, mem_alloc: {}".format(message, steps, unscaled_loss, total_loss / steps,torch.cuda.max_memory_allocated()))

        if training:
            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                loss.backward()
                model.optim.step()
                
        #if training == False: #validation early-stop
        #    early_stopping(loss, model, "./")
        #    print("HELLO WORLD")
        #    if early_stopping.early_stop:
        #        print("Early stopping")
        #        break        
        
    return preds, trues


def preform_experiment(args):
    
    early_stopping = EarlyStopping(verbose=True) #7 as default

    model = get_model(args)
    params = list(get_params(model))
    print('Number of parameters: {}'.format(len(params)))
    for p in params:
        print(p.shape)

    if args.deepspeed:
        deepspeed_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                              model=model,
                                                              model_parameters=params)
    else:
        model.to('cuda')
        model.optim = Adam(params, lr=0.001)

    train_data, train_loader = _get_data(args, flag='train')
    assert len(train_data.data_x[0]) == args.input_len, \
        "Dataset contains input vectors of length {} while input_len is set to {}".format(len(train_data.data_x[0], args.input_len))
    assert len(train_data.data_y[0]) == args.output_len, \
        "Dataset contains output vectors of length {} while output_len is set to {}".format(
            len(train_data.data_y[0]), args.output_len)

    start = time.time()
    for iter in range(1, args.iterations + 1):
        preds, trues = run_iteration(deepspeed_engine if args.deepspeed else model , train_loader, args, training=True, message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        mse, mae = run_metrics("Loss after iteration {}".format(iter), preds, trues)
        if args.local_rank == 0:
            ipc.sendPartials(iter, mse, mae)
        print("Time per iteration {}, memory {}".format((time.time() - start)/iter, torch.cuda.memory_stats()))
        early_stopping(mse, model, "./")
        if early_stopping.early_stop:
                print("Early stopping")
                break

    print(torch.cuda.max_memory_allocated())

    if args.debug:
        model.record()


    test_data, test_loader = _get_data(args, flag='test')
    if deepspeed:
        model.inference()
    else:
        model.eval()
    # Model evaluation on validation data
    v_preds, v_trues = run_iteration(deepspeed_engine if args.deepspeed else model, test_loader, args, training=False, message="Validation set")
    mse, mae = run_metrics("Loss for validation set ", v_preds, v_trues)

    # Send results / plot models if debug option is on
    if args.local_rank == 0:
        ipc.sendResults(mse, mae)
        if args.debug:
            plot_model(args, model)
     
    #Plot prediction + truth data.
    
    #Validation content
    print("Feature count below")
    print(len(v_preds[0][0][0])) #7 features
    print("Timestep count below")
    print(len(v_preds[0][0])) #24 timesteps
    print("Batch_size below")
    print(len(v_preds[0])) #32 batch_size
    
    
        #Get prediction values
    featurepred1 = []
    featurepred2 = []
    featurepred3 = []
    #featurepred4 = []
    #featurepred5 = []
    #featurepred6 = []
    #featurepred7 = []
    for feature in v_preds[0][0]:
      featurepred1.append(feature[0])
      featurepred2.append(feature[1])
      featurepred3.append(feature[2])
      #featurepred4.append(feature[3])
      #featurepred5.append(feature[4])
      #featurepred6.append(feature[5])
      #featurepred7.append(feature[6])
    #print("prediction" + str(featurepred1))

    #Get truth values
    featuretrue1 = []
    featuretrue2 = []
    featuretrue3 = []
    #featuretrue4 = []
    #featuretrue5 = []
    #featuretrue6 = []
    #featuretrue7 = []
    for feature in v_trues[0][0]:
      featuretrue1.append(feature[0])
      featuretrue2.append(feature[1])
      featuretrue3.append(feature[2])
      #featuretrue4.append(feature[3])
      #featuretrue5.append(feature[4])
      #featuretrue6.append(feature[5])
      #featuretrue7.append(feature[6])
    #print("truth" + str(featuretrue1))

    #Plot values       
    x = [1, 2, 3, 4, 5, 6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
   
    y = featurepred1 
    z = featuretrue1
   
    a = featurepred2 
    b = featuretrue2

    c = featurepred3 
    d = featuretrue3

    #e = featurepred4
    #f = featuretrue4

    #g = featurepred5 
    #h = featuretrue5

    #i = featurepred6 
    #j = featuretrue6

    #k = featurepred7 
    #l = featuretrue7

    #Plot Feature 1
    plt.plot(x, y, label='Prediction')
    plt.plot(x, z, label='Truth')

    #Plot Feature 2
    #plt.plot(x, a, label='P_F2')
    #plt.plot(x, b, label='T_F2')

    #Plot Feature 3
    #plt.plot(x, c, label='P_F3')
    #plt.plot(x, d, label='T_F3')

    #Plot Feature 3
    #plt.plot(x, e, label='P_F4')
    #plt.plot(x, f, label='T_F4')

    #Plot Feature 3
    #plt.plot(x, g, label='P_F5')
    #plt.plot(x, h, label='T_F5')

    #Plot Feature 3
    #plt.plot(x, i, label='P_F6')
    #plt.plot(x, j, label='T_F6')

    #Plot Feature 3
    #plt.plot(x, k, label='P_F7')
    #plt.plot(x, l, label='T_F7')

    #Map titles, label and legends
    plt.title('NP15')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('NP15.png')
    plt.close()

    plt.plot(x, a, label='Prediction')
    plt.plot(x, b, label='Truth')

    plt.title('SP15')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('SP15.png')
    plt.close()

    plt.plot(x, c, label='Prediction')
    plt.plot(x, d, label='Truth')

    plt.title('ZP26')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('ZP26.png')
    plt.close()



    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    axes[0].plot(x, y)
    axes[0].plot(x, z)
    axes[1].plot(x, a)
    axes[1].plot(x, b)
    axes[2].plot(x, c)
    axes[2].plot(x, d)
    fig.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    parser = build_parser()
    args = parser.parse_args(None)
    preform_experiment(args)


if __name__ == '__main__':
    main()


