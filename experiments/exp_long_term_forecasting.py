from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, metric_extended
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

torch.autograd.set_detect_anomaly(True)
plt.switch_backend('agg')

# this function added
def save_prediction_grid(examples, name, title, caption, max_examples=16):
    """Save a combined grid of lookback/forecast/prediction windows as a PNG."""
    if not examples:
        return

    n_examples = min(len(examples), max_examples)
    n_cols = 4
    n_rows = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    flat_axes = axes.flatten()
    for idx, sample in enumerate(examples[:n_examples]):
        ax = flat_axes[idx]
        lookback = np.asarray(sample["lookback"])
        forecast = np.asarray(sample["forecast"])
        predicted = np.asarray(sample["predicted"])

        lookback_x = np.arange(len(lookback))
        forecast_x = np.arange(len(lookback), len(lookback) + len(forecast))

        ax.plot(lookback_x, lookback, color='#1f77b4', linewidth=1.8, label='Lookback Window')
        ax.plot(forecast_x, forecast, color='#ff7f0e', linewidth=1.8, label='Forecast Window')
        ax.plot(
            forecast_x,
            predicted,
            color='#d62728',
            linewidth=1.8,
            linestyle='--',
            label='Predicted Window',
        )
        ax.set_title(f'Example {idx + 1}', fontsize=10)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Values')
        ax.grid(True, alpha=0.35)
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    for ax in flat_axes[n_examples:]:
        ax.axis('off')

    fig.suptitle(title, fontsize=15, y=0.985)
    fig.text(0.5, 0.015, caption, ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    fig.savefig(name, dpi=200, bbox_inches='tight')
    plt.close(fig)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)        #wrapper for using multiple gpu
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()  #Mean Absolute Error
        else:
            criterion = nn.MSELoss() #Mean Squared Error
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():    #Disables gradient computation.
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() #creates a tensor of zeros that has the same shape as the future part of batch_y.
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) #cat is concatinate

                if self.args.use_amp:        #Automatic Mixed Precision (AMP) is a technique in deep learning that uses both 32-bit and 16-bit floating point numbers during training to make training faster and use less GPU memory, while keeping model accuracy.
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:] # in this  --self.args.pred_len is done only for safty
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu() #detach() is used to remove the tensor from the computation graph.
                true = batch_y.detach().cpu() #cpu() is used to move the tensor from the GPU to the CPU.

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.numpy()
                    true = true.numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):      #start again from here
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # # Efficiency: dynamic memory footprint
        # # Track dynamic memory usage over an epoch
        # torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) 
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0                        
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y) + self.args.l1_weight * attn[0] 
                    train_loss.append(loss.item())

                if (i + 1) % 30 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    # # Efficiency: dynamic memory footprint
                    # # Record current and peak memory usage after processing this batch
                    # current_memory = torch.cuda.memory_allocated()
                    # peak_memory = torch.cuda.max_memory_allocated()
                    # print(f"Current memory: {current_memory / (1024 ** 2):.2f} MB, Peak memory: {peak_memory / (1024 ** 2):.2f} MB")

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, printout=False)
                    scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)

        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    #changes made here
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './checkpoints/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        max_prediction_examples = 16
        prediction_examples = []
        num_test_batches = max(1, len(test_loader))
        examples_per_batch = max(1, int(np.ceil(max_prediction_examples / num_test_batches)))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)

                if i % 20 == 0:
                    lookback = input[0, :, -1]
                    forecast = true[0, :, -1]
                    predicted = pred[0, :, -1]
                    gt = np.concatenate((lookback, forecast), axis=0)
                    pd = np.concatenate((lookback, predicted), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if len(prediction_examples) < max_prediction_examples:
                    batch_size = pred.shape[0]
                    sample_count = min(
                        examples_per_batch,
                        batch_size,
                        max_prediction_examples - len(prediction_examples),
                    )
                    sample_indices = np.linspace(
                        0,
                        batch_size - 1,
                        num=sample_count,
                        dtype=int,
                    )
                    for sample_idx in sample_indices:
                        lookback = input[sample_idx, :, -1]
                        forecast = true[sample_idx, :, -1]
                        predicted = pred[sample_idx, :, -1]
                        gt = np.concatenate((lookback, forecast), axis=0)
                        pd = np.concatenate((lookback, predicted), axis=0)
                        sample_pdf = os.path.join(folder_path, f'example_{len(prediction_examples):02d}.pdf')
                        visual(gt, pd, sample_pdf)
                        prediction_examples.append(
                            {
                                'lookback': lookback,
                                'forecast': forecast,
                                'predicted': predicted,
                            }
                        )
                        if len(prediction_examples) >= max_prediction_examples:
                            break

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # result save
        folder_path = './checkpoints/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        metrics = metric_extended(preds, trues)
        print(f'Collected {len(prediction_examples)} examples for combined_prediction_examples.png')
        print('mse:{}, mae:{}'.format(metrics['mse'], metrics['mae']))
        print('rmse:{}, mape:{}, mspe:{}'.format(metrics['rmse'], metrics['mape'], metrics['mspe']))
        print(
            'rse:{}, corr:{}, smape:{}, wape:{}, r2:{}'.format(
                metrics['rse'],
                metrics['corr'],
                metrics['smape'],
                metrics['wape'],
                metrics['r2'],
            )
        )
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        metric_parts = [f'{name}:{value}' for name, value in metrics.items()]
        f.write(', '.join(metric_parts))
        f.write('\n')
        f.write('\n')
        f.close()

        dataset_label = self.args.data
        if self.args.data in {'custom', 'PEMS', 'Solar'}:
            dataset_label = os.path.splitext(os.path.basename(self.args.data_path))[0]
        attention_label = getattr(self.args, 'attention_mode', 'original')
        title = (
            f'{self.args.model} ({attention_label}) on {dataset_label}: '
            f'{self.args.seq_len}-step input and {self.args.pred_len}-step predictions'
        )
        caption = (
            'Combined forecasting examples from the test split. '
            'Blue: lookback window. Orange: forecast window. '
            'Red dashed: predicted window. Each subplot shows the last variate '
            'from one sampled test window.'
        )
        save_prediction_grid(
            prediction_examples,
            os.path.join(folder_path, 'combined_prediction_examples.png'),
            title,
            caption,
            max_examples=max_prediction_examples,
        )

    
        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

    
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
