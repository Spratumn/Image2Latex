import torch
from tqdm import tqdm

from utils.loss import LatexLoss
from config import Config
import cv2
cfg = Config()


class Trainer:
    def __init__(self, model, optimizer):
        self.optimizer = optimizer
        self.model = model
        self.loss_stats = {'total_loss': []}
        self.loss = LatexLoss()

    def set_device(self, device):

        self.model = self.model.to(device)
        self.loss = self.loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, epoch, phase, data_loader, log_file):
        epoch_total_loss = 0.0

        if phase == 'train':
            self.model.train()
            self.loss.train()
        else:
            if len(cfg.GPU) > 1:
                self.model = self.model.module
                self.loss = self.loss.module
            self.model.eval()
            self.loss.eval()
            # release cuda cache
            torch.cuda.empty_cache()

        data_process = tqdm(data_loader)
        for batch_item in data_process:
            batch_img, batch_formulas = batch_item
            batch_img = batch_img.to(device=cfg.DEVICE)
            batch_formulas = batch_formulas.to(device=cfg.DEVICE)

            batch_predict = self.model(batch_img)
            loss, loss_stats = self.loss(batch_formulas, batch_formulas)
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_loss = [loss_stats['total_loss']]

            epoch_total_loss += batch_loss[0]

            loss_str = "total_loss: {}"\
                .format(batch_loss[0])

            data_process.set_description_str("epoch:{}".format(epoch))
            data_process.set_postfix_str(loss_str)

        log_str = "{},{:.4f}\n".format(epoch, epoch_total_loss / len(data_loader))
        log_file.write(log_str)
        log_file.flush()

    def val(self, epoch, data_loader, eval_log):
        return self.run_epoch(epoch, 'eval', data_loader, eval_log)

    def train(self, epoch, data_loader, train_log):
        return self.run_epoch(epoch, 'train', data_loader, train_log)







