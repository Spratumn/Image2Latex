# coding:utf-8
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import math
import torch

from utils.tokens import tokenlist
from config import Config

cfg = Config()


class LabelProcess:
	def __init__(self):
		self.tokenlist = tokenlist
		self.tokenlist.sort(key=lambda i: len(i), reverse=True)
		self.token_nums = 4
		self.str2token = {'<f>': 0, '</f>': 1, '<pad>': 2, '<unk>': 3}
		self.token2str = {0: '<f>', 1: '</f>', 2: '<pad>', 3: '<unk>'}
		self.get_token_str_dict()

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
		i = 4
		for str in self.tokenlist:
			if str in self.str2token:
				print(str)
			else:
				self.str2token[str] = i
				self.token2str[i] = str
				i += 1
		self.token_nums = i

	def formula2tokens(self, formula):
		# print(formula)
		matched_formula = []
		matched_index = []
		for i in range(4, self.token_nums):
			str = self.token2str[i]
			res = re.finditer(str, formula)
			res_inds = [m.span() for m in res]
			if len(res_inds) > 0:
				for res_ind in res_inds:
					if res_ind[0] not in matched_index:
						matched_formula.append([res_ind[0], i])
						for idx in range(res_ind[0], res_ind[1]):
							matched_index.append(idx)
		matched_formula = np.array(matched_formula)
		matched_formula = matched_formula[np.argsort(matched_formula[:, 0])]
		tokens = list(matched_formula.transpose()[1])
		post_token_count = cfg.MAX_FORMULA_LENGTH - len(tokens)
		if post_token_count > 0:
			post_token = [2] * post_token_count
			# post_token.append(1)
			tokens = tokens + post_token
		else:
			tokens = tokens[:cfg.MAX_FORMULA_LENGTH]
		return tokens

	def tokens2formula(self, tokens):
		formula = ''
		print(tokens)
		for token in tokens:
			str = self.token2str[token] + ' '
			if str.startswith(r'\\') or str.startswith('\\'):
				str = str[1:-1]
			formula += str
		return formula


if __name__ == '__main__':
	label_p = LabelProcess()
	print(label_p.token_nums)