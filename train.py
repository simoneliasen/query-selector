
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import deepspeed
import torch.nn as nn

from settings.config import build_parser
from model import Transformer
from data.data_loader import Dataset_ETT_hour
from utils.tools import metric, EarlyStopping, visualize_loss, visualize_predictions


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, enc_attn_type=args.encoder_attention,
                       dec_attn_type=args.decoder_attention, dropout=args.dropout)

def get_params(mdl):
    return mdl.parameters()

def _get_data(args, flag):
    #Not only applicable for ETT
    Data = Dataset_ETT_hour

    if flag == 'test':
        shuffle_flag = False;
        drop_last = True;
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size

    data_set = Data(
        root_path='data',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
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
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}'.format(caption, mse, mae))
    return mse, mae


#Only current differenee to validate() is loading of checkpoint
def test(args, model, deepspeed_engine):
    best_model_path = 'checkpoints/checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path)) 

    if args.debug:
        model.record()

    test_data, test_loader = _get_data(args, flag='test')

    if deepspeed:
        model.inference()
    else:
        model.eval()

    v_preds, v_trues = run_iteration(deepspeed_engine if args.deepspeed else model, test_loader, args, training=False, message="Validation set")
    mse, mae = run_metrics("Loss for validation set ", v_preds, v_trues)

    visualize_predictions(v_preds,v_trues)

    #Set back to training (does not handle model.inference)
    model.train()

    return mse, mae 


def validate(args, model, deepspeed_engine):
    
    if args.debug:
        model.record()

    test_data, test_loader = _get_data(args, flag='test')

    if deepspeed:
        model.inference()
    else:
        model.eval()

    v_preds, v_trues = run_iteration(deepspeed_engine if args.deepspeed else model, test_loader, args, training=False, message="Validation set")
    mse, mae = run_metrics("Loss for validation set ", v_preds, v_trues)

    visualize_predictions(v_preds,v_trues)

    #Set back to training (does not handle model.inference)
    model.train()

    return mse, mae 


def run_iteration(model, loader, args, training=True, message = ''):
    preds = []
    trues = []
    total_loss = 0
    elem_num = 0
    steps = 0
    target_device = 'cuda:{}'.format(args.local_rank)
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if not args.deepspeed:
            model.optim.zero_grad()

        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)

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

    return preds, trues


def preform_experiment(args):
    early_stopping = EarlyStopping(patience=10, verbose=True) #7 as default, set to 3 for testing

    model = get_model(args)
    params = list(get_params(model))

    if args.deepspeed:
        deepspeed_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=params)
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

    train_mses = []
    train_maes = []
    val_mses = []
    val_maes = []

    for iter in range(1, args.iterations + 1):
        preds, trues = run_iteration(deepspeed_engine if args.deepspeed else model , train_loader, args, training=True, message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        train_mse, train_mae = run_metrics("Loss after iteration {}".format(iter), preds, trues)
        train_mses.append(train_mse)
        train_maes.append(train_mae)
        
        if iter % 3 == 0:
          val_mse, val_mae = validate(args, model, deepspeed_engine)
          val_mses.append(val_mse)
          val_maes.append(val_mae)
          early_stopping(val_mae, model, "./")
          if early_stopping.early_stop:
            print("Early stopping")
            break
          
    visualize_loss(train_mses, train_maes, val_mses, val_maes)

    print(torch.cuda.max_memory_allocated())


def main():
    parser = build_parser()
    args = parser.parse_args(None)
    preform_experiment(args)

if __name__ == '__main__':
    main()