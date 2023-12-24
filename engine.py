import sys
import copy
import torch


class Engine:
    def __init__(self, data, model, loss_func, optimizer, scheduler, num_epochs):
        self.data = data
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

    def train(self):
        print("start training...", flush=True)
        min_valid_epoch = 0
        min_valid_rmse = sys.maxsize
        min_valid_mae = sys.maxsize
        min_valid_mape = sys.maxsize
        min_valid_smape = sys.maxsize
        for epoch in range(self.num_epochs):
            self.model.train()
            for iter, (seq, label) in enumerate(self.data.train_loader):
                seq = self.data.scaler.transform(seq)
                output = self.model(seq)
                output = self.data.scaler.inverse_transform(output)
                output = torch.nan_to_num(output, nan=0.0)
                loss = self.loss_func(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                if iter % 50 == 0:
                    rmse, mae, mape, smape = self.metric(output, label)
                    log = 'Iter: {:03d}, Train RMSE: {:.3f}, Train MAE: {:.3f}, Train MAPE: {:.3f}, Train SMAPE: {:.3f}'
                    print(log.format(iter, rmse, mae, mape, smape), flush=True)
            self.model.eval()
            for seq, label in self.data.valid_loader:
                seq = self.data.scaler.transform(seq)
                with torch.no_grad():
                    output = self.model(seq)
                output = self.data.scaler.inverse_transform(output)
                rmse, mae, mape, smape = self.metric(output, label)
                log = 'Epoch: {:03d}, Valid RMSE: {:.3f}, Valid MAE: {:.3f}, Valid MAPE: {:.3f}, Valid SMAPE: {:.3f}\n'
                print(log.format(epoch + 1, rmse, mae, mape, smape, flush=True))
                if rmse < min_valid_rmse:
                    min_valid_epoch = epoch
                    min_valid_rmse = rmse
                    min_valid_mae = mae
                    min_valid_mape = mape
                    min_valid_smape = smape
                    best_model = copy.deepcopy(self.model)
            self.scheduler.step()

        log = 'Evaluate best model on valid data for epoch {:d}, Valid RMSE: {:.3f}, Valid MAE: {:.3f}, Valid MAPE: {:.3f}, Valid SMAPE: {:.3f}'
        print(log.format(min_valid_epoch + 1, min_valid_rmse, min_valid_mae, min_valid_mape, min_valid_smape))
        torch.save({'model_state_dict': best_model.state_dict()}, 'checkpoints/best_model.pt')

    def test(self):
        self.model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
        for seq, label in self.data.test_loader:
            seq = self.data.scaler.transform(seq)
            with torch.no_grad():
                output = self.model(seq)
            output = self.data.scaler.inverse_transform(output)
            output = torch.nan_to_num(output, nan=0.0)
            rmse, mae, mape, smape = self.metric(output, label)
            log = 'Evaluate best model on test data, Test RMSE: {:.3f}, Test MAE: {:.3f}, Test MAPE: {:.3f}, Test SMAPE: {:.3f}'
            print(log.format(rmse, mae, mape, smape))

    @staticmethod
    def metric(output, label):
        rmse = torch.sqrt(torch.mean(torch.square(label - output)))
        mae = torch.mean(torch.abs(label - output))
        mape = torch.mean(torch.abs((label - output) / label)) * 100
        smape = torch.mean(torch.abs((label - output) / (torch.abs(label) + torch.abs(output)) / 2)) * 100
        return rmse.item(), mae.item(), mape.item(), smape.item()
