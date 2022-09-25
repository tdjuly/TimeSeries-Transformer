import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import chain
import datetime

from TimeSeriesTransformer import TimeSeriesTransformer
from get_src_trg import get_src_trg
from generate_square_subsequent_mask import generate_square_subsequent_mask
from accuracy import get_mape, get_rmse
from MyDataset import MyDataset

if __name__ == '__main__':
    num_layer = [1, 2, 3, 4]
    port_name = []
    output = []
    for j in range(4):
        print(j)

        # Define some hyperparameters of model
        dim_val = 512  # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
        input_size = 1  # The number of input variables. 1 if univariate forecasting.
        dec_seq_len = 3  # length of input given to decoder. Can have any integer value.
        max_seq_len = 3  # What's the longest sequence the model will encounter? Used to make the positional encoder
        target_seq_len = 1  # Length of the target sequence, i.e. how many time steps should your forecast cover
        n_encoder_layers = num_layer[j]  # Number of times the encoder layer is stacked in the encoder
        n_decoder_layers = num_layer[j]  # Number of times the decoder layer is stacked in the decoder
        n_heads = 8  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
        batch_size = 1
        enc_seq_len = dec_seq_len  # length of input given to encoder. Can have any integer value.

        epochs = 1000
        lr = 0.0001
        weight_decay = 0.0001
        ratio = 0.8

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------------data processing-----------------------------------
        path = os.path.dirname(os.path.realpath(__file__)) + '/data/port_data.csv'
        df = pd.read_csv(path, encoding='gbk')
        df.fillna(df.mean(), inplace=True)
        columns = df.columns
        for jj in range(len(df.columns)-1):
            print('\t', columns[jj + 1])

            src_trg_trgy = []
            df_1 = pd.DataFrame(df.iloc[:, jj+1])
            _max = np.asarray(np.max(df_1))
            _min = np.asarray(np.min(df_1))
            df_1 = (df_1 - _min) / (_max - _min)

            for i in range(len(df_1) - dec_seq_len):
                src = df_1.iloc[i:i + dec_seq_len, 0]
                trg = df_1.iloc[i + dec_seq_len - 1:i + dec_seq_len - 1 + target_seq_len, 0]
                trg_y = df_1.iloc[i + dec_seq_len:i + dec_seq_len + target_seq_len, 0]

                src = torch.FloatTensor(np.asarray(src))
                trg = torch.FloatTensor(np.asarray(trg))
                trg_y = torch.FloatTensor(np.asarray(trg_y))

                src_trg_trgy.append((src, trg, trg_y))

            train_len = int(int(len(src_trg_trgy)) * ratio)
            test_len = len(src_trg_trgy) - train_len

            train_set = src_trg_trgy[0:train_len]
            test_set = src_trg_trgy

            train = MyDataset(train_set)
            test = MyDataset(test_set)

            train_set = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
            test_set = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

            src_mask = generate_square_subsequent_mask(dim1=batch_size * n_heads, dim2=target_seq_len, dim3=enc_seq_len)

            # Make tgt mask for decoder with size:
            # [batch_size*n_heads, target_seq_len, target_seq_len]
            tgt_mask = generate_square_subsequent_mask(dim1=batch_size * n_heads, dim2=target_seq_len, dim3=target_seq_len)

            # -----------------------------------build model---------------------------------------------------
            model = TimeSeriesTransformer(
                dim_val=dim_val,
                input_size=input_size,
                dec_seq_len=dec_seq_len,
                max_seq_len=max_seq_len,
                out_seq_len=target_seq_len,
                n_decoder_layers=n_decoder_layers,
                n_encoder_layers=n_encoder_layers,
                n_heads=n_heads
            )

            loss_function = nn.MSELoss().to(device)

            # define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # -----------------------------------training---------------------------------------------------
            loss = 0
            best_val_loss = float("inf")
            best_model = None
            best_optimizer = None
            for i in range(epochs):
                for (src, trg, trg_y) in train_set:
                    src = torch.reshape(src, (dec_seq_len, 1))
                    src = src.to(device)
                    trg_y = trg_y.to(device)
                    y_pred = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)
                    loss = loss_function(y_pred.reshape(1, 1), trg_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_model = model
                    best_optimizer = optimizer
                # print('epoch:', i, '\tMES:', loss.item())

            T = datetime.datetime.now()
            model_dir = r'./model/time-series-transformer.pkl' + str(T)
            state = {'model': best_model.state_dict(), 'optimizer': best_optimizer.state_dict()}
            # print("\tbest model saved, best loss is:", best_val_loss)
            torch.save(state, model_dir)

            model = TimeSeriesTransformer(
                dim_val=dim_val,
                input_size=input_size,
                dec_seq_len=dec_seq_len,
                max_seq_len=max_seq_len,
                out_seq_len=target_seq_len,
                n_decoder_layers=n_decoder_layers,
                n_encoder_layers=n_encoder_layers,
                n_heads=n_heads
            )
            model.load_state_dict(torch.load(model_dir)['model'])
            model.eval()

            # ------------------------------------------ testing----------------------------------------------
            pred = []
            y = []
            for (src, trg, trg_y) in test_set:
                trg_y = list(chain.from_iterable(trg_y.data.tolist()))
                y.extend(trg_y)
                src = torch.reshape(src, (dec_seq_len, 1))
                src = src.to(device)
                with torch.no_grad():
                    y_pred = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)
                    y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                    pred.extend(y_pred)

            y, pred = np.array([y]), np.array([pred])
            y = (_max - _min) * y + _min
            y = y.flatten()
            pred = (_max - _min) * pred + _min
            pred = pred.flatten()

            # train fitted
            train_fitted = pred[:train_len]
            train_real = y[:train_len]
            train_MAPE = get_mape(train_real, train_fitted)
            train_RMSE = get_rmse(train_real, train_fitted)

            # test forecast results
            test_results = pred[train_len:]
            test_real = y[train_len:]
            test_MAPE = get_mape(test_real, test_results)
            test_RMSE = get_rmse(test_real, test_results)
            # print('\ntest real: ', test_real)
            # print("test forecasted: ", test_results)
            # print('test_MAPE:', test_MAPE)
            # print('test_RMSE:', test_RMSE)

            _output = [n_encoder_layers, columns[jj+1], train_fitted, train_real, train_MAPE, train_RMSE, test_results, test_real, test_MAPE, test_RMSE]
            output.append(_output)

    # # --------------------------------save results--------------------------------
    import csv
    f = open('output.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    header = ('num_layer', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
    csv_writer.writerow(header)
    for data in output:
        csv_writer.writerow(data)
    f.close()
