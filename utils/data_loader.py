import os
import glob
import cv2
import pickle
import numpy as np
from collections import defaultdict

im2latex100k_path="../../../week20/image2latex100k/"
class Vocab:
    def __init__(self):
        self.pathes = im2latex100k_path+"im2latex_formulas.lst"
        self.unk_threshold = 10   # 在is_eligible函数中，判断是否保存倒token表中
        self.token2idx = {}
        self.idx2token = {}
        self.load()

    def build(self):
        self.start_token = 0
        self.end_token = 1
        self.pad_token = 2
        self.unk_token = 3
        self.frequency = defaultdict(int)
        self.total = 0
        for path in self.pathes:
            formulas = open(path, 'r',encoding="latin_1")
            lines = formulas.readlines()
            for line in lines:
                tokens = line.rstrip('\n').strip(' ').split()
                for token in tokens:
                    self.frequency[token] += 1
                    self.total += 1
        self.token2idx = {'<f>' : 0, '</f>' : 1, '<pad>' : 2, '<unk>' : 3}
        self.idx2token = {0 : '<f>', 1 : '</f>', 2 : '<pad>', 3 : '<unk>'}
        idx = 4
        for path in self.pathes:
            formulas = open(path, 'r')
            lines = formulas.readlines()
            for line in lines:
                tokens = line.rstrip('\n').strip(' ').split()
                for token in tokens:
                    if self.is_eligible(token) and token not in self.token2idx:
                        self.token2idx[token] = idx
                        self.idx2token[idx] = token
                        idx += 1
        # save vocab
        if not os.path.isdir('vocab'):
            os.mkdir('vocab')
        f = open(os.path.join('vocab', 'vocab.pkl'), 'wb')
        pickle.dump(self, f)
        f.close()

    def is_eligible(self, token):   #出现次数与阈值相比较，决定是否保存
        if self.frequency[token] >= self.unk_threshold:
            return True
        return False

    def load(self):
        try:
            with open(os.path.join('vocab', 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
                self.token2idx = vocab.token2idx
                self.idx2token = vocab.idx2token
                self.start_token = vocab.start_token
                self.unk_token = vocab.unk_token
                self.pad_token = vocab.pad_token
                self.end_token = vocab.end_token
                self.frequency = vocab.frequency
                self.total = vocab.total
        except:
            self.build()

    def formulas2tensor(self, formulas, max_len):
        '''
	week20作业
        input: formulas，若干个公式字符串
	output: tensor，若谷个公式对应得序号列表
	'''
         	
        return tensor

    def tensor2formula(self, tensor, pretty=False, tags=True):
        if not pretty:
            if tags:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            else:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0])
                                if self.idx2token[tensor[i]] not in ['<f>', '</f>', '<pad>'])
        else:
            s = ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            end = s.find('</f>')
            if end != -1 : end = end - 1
            s = s[4:end]
            s = s.replace('<pad>', '')
            s = s.replace('<unk>', '')
            return s

def get_train_data(formulas_path,self.train_path,self.images_path):
    images=[]
    formulas=[]
    return images,formulas


class data_loader:
    def __init__(self, vocab):
        '''
            input: formula_vocab
            output: train/validate: batch_imgs, batch_formula_tensor, end_of_epoch
                    test: batch_imgs, end_of_epoch
        '''
        self.vocab = vocab
        self.batch_size =24
        self.sort_by_formulas_len = False
        self.shuffle = False
        self.cursor = 0  #用来计算当前读取的图片数量，若epoch结束，reset (↓def move_cursor)
        self.formulas_path=im2latex100k_path+"/im2latex_formulas.lst"
        #self.train_file_path=im2latex100k_path+"/im2latex_validate.lst"
        #self.train_file_path=im2latex100k_path+"/im2latex_test.lst"
        self.train_file_path=im2latex100k_path+"/im2latex_train.lst"
        self.images_path=im2latex100k_path+"/formula_images/"
	# 根据readme中的对应关系，将图片和公式成对读出，放到self.image中，sefl.formulas中
	# 实现get_train_data函数
        self.images,self.formulas=get_train_data(self.formulas_path,self.train_file_path,self.images_path)
        self.has_label = (self.formulas_path is not None)

    def get_next_batch(self):
        current_batch_size = min(self.batch_size, len(self.images) - self.cursor)  #比较当前剩余size与batch_size大小，得到当前的batch_size
        if current_batch_size == 0: end_of_epoch = True
        if self.has_label:
            if self.sort_by_formulas_len:
                max_batch_len = len(self.formulas[self.cursor])
            else:
                max_batch_len = -1
                for i in range(self.cursor, self.cursor + current_batch_size):
                    max_batch_len = max(max_batch_len, len(self.formulas[i]))
            batch_formulas_tensor = self.vocab.formulas2tensor(
                self.formulas[self.cursor:self.cursor + current_batch_size], max_batch_len)
            batch_formulas=self.formulas[self.cursor:self.cursor + current_batch_size]

        # 读取images，存到batch_imgs中
        batch_imgs = []
        for i in range(current_batch_size):
            img = cv2.imread(self.images[self.cursor], cv2.IMREAD_GRAYSCALE)
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            img = self.normalize(img)
            batch_imgs.append(img)
            end_of_epoch = self.move_cursor()
        if self.has_label:
            return np.array(batch_imgs), batch_formulas_tensor,batch_formulas,end_of_epoch
        else:
            return np.array(batch_imgs), end_of_epoch

    def move_cursor(self): #计算当前图片数量
        self.cursor += 1
        if len(self.images) <= self.cursor:
            self.reset_cursor()
            return True
        return False

    def normalize(self, image): #标准化
        return (image / 255.) * 2 - 1

    def reset_cursor(self): #重置cursor
        self.cursor = 0

if __name__ == '__main__':
    #建立字典
    vocab = Vocab()
    print('Data loading... ')
    #用字典进行编码
    data = data_loader(vocab)
    image_train, formula_train, epoch_ended = data_loader.get_next_batch(data)
    print('images_size:', image_train.shape)
    print('formulas_size:', formula_train.shape)
    print('formula:', formula_train)
