# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import torch

from config import Config

cfg = Config()


class LabelProcess:
	def __init__(self):
		self.token2str = {'<f>': 0, '</f>': 1, '<pad>': 2, '<unk>': 3}
		self.str2token = {0: '<f>', 1: '</f>', 2: '<pad>', 3: '<unk>'}

	def formulas2tensor(self, formulas):
		"""
		input: formulas，若干个公式字符串
		output: tensor，若谷个公式对应得序号列表
		"""

		return

	def tensor2formula(self, tensor, pretty=False, tags=True):
		"""
		input: tensor，若谷个公式对应得序号列表
		output: formulas，若干个公式字符串

		"""
		return

	def get_token_str_dict(self):
		with open(cfg.FORMULAS_DIR, 'r', encoding="latin_1") as f:
			formulas_lines = f.readlines()
		# for formulas_line in formulas_lines:
		# 	for s in



