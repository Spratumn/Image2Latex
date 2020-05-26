import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.resnet import res_34
from config import Config
cfg = Config()


def create_model():
    model = LatexNet()
    return model


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model


def save_model(path, epoch, model):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    torch.save(data, path)


class LatexNet(nn.Module):
    def __init__(self):
        super(LatexNet, self).__init__()
        # features: [[64,1/4], [128,1/8], [256,1/16], [512,1/32]]
        self.backbone = res_34(load_weight=True, output_all_layer=True)

        # upsample to 1/16, make use of last two resnet output layers
        self.merger = MergeModel()

        self.encoder = Encoder()  #
        self.decoder = Decoder()  #

        self.formulas = ["" for i in range(cfg.BATCH_SIZE)]
        self.logits = []
        self.predicts = []

    def forward(self, x):
        features = self.backbone(x)
        feature = self.merger(features)
        seq_features, (ehn, ecn) = self.encoder(feature)

        token_start = torch.tensor([0 for i in range(cfg.BATCH_SIZE)])
        token_vector = torch.nn.functional.one_hot(token_start, cfg.INPUT_SIZE_DECODER)
        # for i in range(0, cfg.BATCH_SIZE):
        #     token_numpy = token_start.cpu().numpy()
        #     self.formulas[i] += token_numpy[i]
        #     # formulas[i] += vocab.idx2token[token_numpy[i]]

        for t in range(1, cfg.SEQUENCE_NUM):
            predict = self.decoder(token_vector, ehn, ecn)
            self.logits.append(predict)
            next_token = predict.argmax(dim=1)
            self.predicts.append(next_token)
            # for i in range(0, cfg.BATCH_SIZE):
            #     token_numpy = next_token.cpu().numpy()
            #     self.formulas[i] += (str(token_numpy[i]) + " ")
            token_vector = torch.nn.functional.one_hot(next_token, cfg.INPUT_SIZE_DECODER)

        return self.predicts


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(cfg.INPUT_SIZE_ENCODER, cfg.HIDDEN_SIZE_ENCODER, batch_first=True)

        self.h0 = torch.randn(1, cfg.BATCH_SIZE, cfg.HIDDEN_SIZE_ENCODER)
        self.c0 = torch.randn(1, cfg.BATCH_SIZE, cfg.HIDDEN_SIZE_ENCODER)

    def forward(self, features):
        # input: features: [n, c, h, w]
        # output: features: [batch_size, seq_size, hidden_size]
        batch_size, channel_size = cfg.BATCH_SIZE, cfg.INPUT_SIZE_ENCODER
        features = features.view(batch_size, channel_size, -1)  # [n, c, h*w]
        input_vector = features.transpose(2, 1)  # [n, h*w, c] <-> [batch_size, seq_size, input_size]
        # seq_features: [batch_size, seq_size, hidden_size], seq_size = down sample w * h
        # import pdb
        # pdb.set_trace()
        seq_features, (hn, cn) = self.lstm(input_vector, (self.h0, self.c0))
        # seq_features = seq_features[:, -1, :]
        return seq_features, (hn, cn)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(cfg.INPUT_SIZE_DECODER, cfg.HIDDEN_SIZE_DECODER, batch_first=True)
        self.out = nn.Linear(cfg.HIDDEN_SIZE_DECODER, cfg.OUTPUT_SIZE_DECODER, bias=False)

    def forward(self, token_vector, hn, cn):
        # token_vector: [batch_size, input_size] -> [batch_size, 1, input_size]
        input_vector = token_vector.unsqueeze(1).float()
        # [batch_size, 1, input_size] -> [batch_size, 1, hidden_size]
        output_vector, (dhn, dcn) = self.lstm(input_vector, (hn, cn))
        output_vector = output_vector.squeeze(1)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        output = self.out(output_vector)
        return output


class MergeModel(nn.Module):
    def __init__(self, layer_config=(256, cfg.INPUT_SIZE_ENCODER)):
        super(MergeModel, self).__init__()
        up_conv_layer_1 = [nn.Conv2d(512, layer_config[0], 1),
                           nn.BatchNorm2d(layer_config[0]),
                           nn.ReLU(),
                           nn.Conv2d(layer_config[0], layer_config[0], 3, padding=1),
                           nn.BatchNorm2d(layer_config[0]),
                           nn.ReLU()]
        up_conv_layer_2 = [nn.Conv2d(256 + layer_config[0], layer_config[1], 1),
                           nn.BatchNorm2d(layer_config[1]),
                           nn.ReLU(),
                           nn.Conv2d(layer_config[1], layer_config[1], 3, padding=1),
                           nn.BatchNorm2d(layer_config[1]),
                           nn.ReLU()]

        self.up_conv_1 = nn.Sequential(*up_conv_layer_1)
        self.up_conv_2 = nn.Sequential(*up_conv_layer_2)

    def forward(self, features):
        # features: [[64,1/4], [128,1/8], [256,1/16], [512,1/32]]
        feature = self.up_conv_1(features[3])  # 512->256
        feature = F.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        feature = torch.cat((feature, features[2]), 1)  # 256+256
        feature = self.up_conv_2(feature)  # 256+256->out_channel
        return feature


