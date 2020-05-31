import torch
import os
import torch.nn as nn
import pdb
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model.model import create_model, load_model, save_model
from utils.dataset import LatexDataset
from utils.trainer import Trainer
from config import Config

cfg = Config()


def train():
	device = torch.device('cuda' if cfg.GPU[0] >= 0 else 'cpu')

	start_epoch = 1
	if start_epoch == 1:
		train_log = open(os.path.join(cfg.LOG_DIR, "train_log.csv"), 'w')
		train_log_title = "epoch,total_loss,classify_loss,angle_loss,iou_loss\n"
		train_log.write(train_log_title)
		train_log.flush()
	else:
		train_log = open(os.path.join(cfg.LOG_DIR, "train_log.csv"), 'a')

	print('Creating model...')
	model = create_model()
	if start_epoch != 1:
		model = load_model(model, 'logs/weights/model_epoch_{}.pth'.format(start_epoch - 1))
	optimizer = torch.optim.Adam(model.parameters(), cfg.LR)

	trainer = Trainer(model, optimizer)
	trainer.set_device(device)
	print('Setting up data...')
	train_loader = DataLoader(LatexDataset(),
							  batch_size=cfg.BATCH_SIZE,
							  shuffle=True,
							  num_workers=cfg.NUM_WORKERS,
							  pin_memory=True,
							  drop_last=True)
	print('Starting training...')
	epoch = start_epoch
	for epoch in range(start_epoch, start_epoch + cfg.EPOCHS):
		trainer.train(epoch, train_loader, train_log)
		if epoch % 5 == 0:
			save_model('logs/weights/model_epoch_{}.pth'.format(epoch), epoch, model)

	save_model(os.path.join(cfg.WEIGHTS_DIR, 'model_last.pth'), epoch, model)


def model_test():
	latnet = create_model()
	img = torch.rand(cfg.BATCH_SIZE, 3, 512, 1600)
	logits, pred = latnet(img)
	print(logits.size())


def dataset_test():
	datst = LatexDataset()
	img, tokens = datst.__getitem__(4341)
	print(tokens)
	print(tokens.size())


if __name__ == '__main__':
	datst = LatexDataset()
	img, tokens = datst.__getitem__(2982)
	img = img.unsqueeze(0)
	latnet = create_model()
	logits, pred = latnet(img)
	pos_logits = torch.gather(logits[0], dim=1, index=tokens)
	loss = -1 * torch.log(pos_logits)
	print(loss.sum())



