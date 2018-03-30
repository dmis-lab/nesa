import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import os
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import profile


class NETS(nn.Module):
    def __init__(self, config, widx2vec, _iter=None):
        super(NETS, self).__init__()
        self.config = config

        self.n_classes = config.slot_size // config.class_div
        self.n_day_slots = self.n_classes // 7

        self.cuda_is_available = torch.cuda.is_available()

        # embedding layers
        self.char_embed = nn.Embedding(config.char_vocab_size,
                                       config.char_embed_dim,
                                       padding_idx=1)
        self.word_embed = nn.Embedding(config.word_vocab_size,
                                       config.word_embed_dim,
                                       padding_idx=1)

        if not config.no_intention or not config.no_snapshot:
            self.user_embed = nn.Embedding(config.user_size,
                                           config.user_embed_dim)
        if not config.no_intention:
            self.dur_embed = nn.Embedding(config.dur_size, config.dur_embed_dim)
        if not config.no_snapshot:
            self.slot_embed = nn.Embedding(self.n_classes,
                                           config.slot_embed_dim)
            self.emtpy_long = Variable(torch.cuda.LongTensor([])
                                       if self.cuda_is_available
                                       else torch.LongTensor([]))

        # dimensions according to settings
        self.num_directions = config.num_directions
        self.t_rnn_idim = config.word_embed_dim + sum(config.tc_conv_fn)
        self.st_rnn_idim = config.word_embed_dim + sum(config.tc_conv_fn)
        self.sm_conv1_idim = config.user_embed_dim + config.slot_embed_dim
        if not config.no_snapshot and not config.no_snapshot_title:
            self.sm_conv1_idim += config.st_rnn_hdim * self.num_directions
            self.empty_st_rnn_output = Variable(torch.zeros(
                    1, self.config.st_rnn_hdim * self.num_directions))
            if self.cuda_is_available:
                self.empty_st_rnn_output = self.empty_st_rnn_output.cuda()
        self.sm_conv2_idim = sum(config.sm_conv_fn[:len(config.sm_conv_fn)//2])

        self.it_idim = config.user_embed_dim + config.dur_embed_dim
        if not config.no_title:
            self.it_idim += config.t_rnn_hdim * self.num_directions
        self.mt_idim = 0
        if not config.no_intention:
            self.mt_idim += self.it_idim
        else:
            if not config.no_title:
                self.mt_idim += config.t_rnn_hdim * self.num_directions
        if not config.no_snapshot:
            self.snapshot_odim = sum(
                config.sm_conv_fn[len(config.sm_conv_fn) // 2:])

            self.mt_idim += config.sm_day_num * config.sm_slot_num
            self.mt_idim += self.snapshot_odim

        # convolution layers
        self.tc_conv = nn.ModuleList(
            [nn.Conv2d(config.char_embed_dim, config.tc_conv_fn[i],
                       (config.tc_conv_fh[i], config.tc_conv_fw[i]),
                       stride=1) for i in range(len(config.tc_conv_fn))])
        self.tc_conv_bn = nn.ModuleList(
            [nn.BatchNorm2d(num_tc_conv_f)
             for num_tc_conv_f in config.tc_conv_fn])
        self.tc_conv_min_dim = len(config.tc_conv_fn) + 1

        if not config.no_snapshot:
            self.sm_conv1 = nn.ModuleList([nn.Conv2d(
                    self.sm_conv1_idim, config.sm_conv_fn[i],
                    (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                    stride=1, padding=config.sm_conv_pd[i])
                    for i in range(0, len(config.sm_conv_fn)//2)])
            self.sm_mp1 = nn.MaxPool2d(2)
            self.sm_conv1_bn = nn.BatchNorm2d(self.sm_conv2_idim)
            self.sm_conv2 = nn.ModuleList([nn.Conv2d(
                    self.sm_conv2_idim,
                    config.sm_conv_fn[i + len(config.sm_conv_fn)//2],
                    (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                    stride=1, padding=config.sm_conv_pd[i])
                    for i in range(len(config.sm_conv_fn)//2)])
            self.sm_mp2 = nn.MaxPool2d(2)
            self.sm_conv2_bn = nn.BatchNorm2d(self.snapshot_odim)

        # rnn layers
        self.batch_first = False
        self.bidirectional = config.num_directions == 2

        if 'lstm' == config.rnn_type:
            if not config.no_title:
                self.t_rnn = nn.LSTM(self.t_rnn_idim, config.t_rnn_hdim,
                                     config.t_rnn_ln,
                                     dropout=config.t_rnn_dr,
                                     batch_first=self.batch_first,
                                     bidirectional=self.bidirectional)

            if not config.no_snapshot and not config.no_snapshot_title:
                self.st_rnn = nn.LSTM(self.st_rnn_idim, config.st_rnn_hdim,
                                      config.st_rnn_ln,
                                      dropout=config.st_rnn_dr,
                                      batch_first=self.batch_first,
                                      bidirectional=self.bidirectional)
        elif 'gru' == config.rnn_type:
            if not config.no_title:
                self.t_rnn = nn.GRU(self.t_rnn_idim, config.t_rnn_hdim,
                                    config.t_rnn_ln,
                                    dropout=config.t_rnn_dr,
                                    batch_first=self.batch_first,
                                    bidirectional=self.bidirectional)
            if not config.no_snapshot and not config.no_snapshot_title:
                self.st_rnn = nn.GRU(self.st_rnn_idim, config.st_rnn_hdim,
                                     config.st_rnn_ln,
                                     dropout=config.st_rnn_dr,
                                     batch_first=self.batch_first,
                                     bidirectional=self.bidirectional)
        else:
            raise ValueError('Invalid RNN: %s' % config.rnn_type)

        # linear layers
        if not config.no_intention:
            self.it_nonl = nn.Linear(self.it_idim, self.it_idim)
            self.it_gate = nn.Linear(self.it_idim, self.it_idim)
        self.mt_nonl = nn.Linear(self.mt_idim, self.mt_idim)
        self.mt_gate = nn.Linear(self.mt_idim, self.mt_idim)
        self.output_fc1 = nn.Linear(self.mt_idim,
                                    config.sm_day_num * config.sm_slot_num)

        # initialization
        self.init_word_embed(widx2vec)
        self.init_convs()
        self.init_linears()

        params = self.model_params(debug=False)
        self.optimizer = optim.Adam(params, lr=config.lr,
                                    weight_decay=config.wd)
        self.criterion = nn.CrossEntropyLoss()

        if config.summary and _iter is not None:
            summary_path = 'runs/' + config.model_name + '_' + str(_iter)
            self.train_writer = SummaryWriter(log_dir=summary_path + '/train')
            self.valid_writer = SummaryWriter(log_dir=summary_path + '/valid')
            self.test_writer = SummaryWriter(log_dir=summary_path + '/test')

    def init_word_embed(self, widx2vec):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(widx2vec)))
        self.word_embed.weight.requires_grad = False

    def init_convs(self):
        def init_conv_list(conv_list):
            for conv in conv_list:
                # https://discuss.pytorch.org/t/weight-initilzation/157/9
                nn.init.xavier_uniform(conv.weight.data)
                nn.init.uniform(conv.bias.data)
        init_conv_list(self.tc_conv)

        if not self.config.no_snapshot:
            init_conv_list(self.sm_conv1)
            init_conv_list(self.sm_conv2)

    def init_linears(self, init='xavier_uniform'):
        # https://github.com/pytorch/pytorch/blob/v0.3.1/torch/nn/modules/linear.py#L48
        def linear_init_uniform(linear, stdv_power=1.):
            stdv = 1. / math.sqrt(linear.weight.size(1))
            stdv *= stdv_power
            nn.init.uniform(linear.weight, -stdv, stdv)
            if linear.bias is not None:
                nn.init.uniform(linear.bias, -stdv, stdv)

        if 'xavier_uniform' == init:
            if not self.config.no_intention:
                nn.init.xavier_uniform(self.it_nonl.weight,
                                       gain=nn.init.calculate_gain('relu'))
                nn.init.uniform(self.it_nonl.bias)
                nn.init.xavier_uniform(self.it_gate.weight, gain=1)
                nn.init.uniform(self.it_gate.bias)
            nn.init.xavier_uniform(self.mt_nonl.weight,
                                   gain=nn.init.calculate_gain('relu'))
            nn.init.uniform(self.mt_nonl.bias)
            nn.init.xavier_uniform(self.mt_gate.weight, gain=1)
            nn.init.uniform(self.mt_gate.bias)
            nn.init.xavier_uniform(self.output_fc1.weight,
                                   gain=nn.init.calculate_gain('relu'))
            nn.init.uniform(self.output_fc1.bias)
        elif 'uniform' == init:
            stdv_pow = 0.5
            if not self.config.no_intention:
                linear_init_uniform(self.it_nonl, stdv_power=stdv_pow)
                linear_init_uniform(self.it_gate, stdv_power=stdv_pow)
            linear_init_uniform(self.mt_nonl, stdv_power=stdv_pow)
            linear_init_uniform(self.mt_gate, stdv_power=stdv_pow)
            linear_init_uniform(self.output_fc1, stdv_power=stdv_pow)

    def init_rnn_h(self, batch_size, rnn_ln, hdim):
        h_0 = Variable(torch.zeros(rnn_ln * self.num_directions,
                                   batch_size, hdim))
        if self.cuda_is_available:
            h_0 = h_0.cuda()

        if 'lstm' == self.config.rnn_type:
            c_0 = Variable(
                        torch.zeros(rnn_ln * self.num_directions,
                                    batch_size, hdim))
            if self.cuda_is_available:
                c_0 = c_0.cuda()
            return h_0, c_0
        elif 'gru' == self.config.rnn_type:
            return h_0
        else:
            raise ValueError('Invalid RNN: %s' % self.config.rnn_type)

    def model_params(self, debug=True):
        print('model parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s' % '{:,}'.format(total_size))
        return params

    def get_rnn_out(self, batch_size, batch_max_seqlen, tl,
                    packed_lstm_input, idx_unsort,
                    rnn, rnn_hdim, rnn_out_dr, rnn_ln):
        assert idx_unsort is not None

        # rnn
        rnn_out, (ht, ct) = rnn(packed_lstm_input,
                                self.init_rnn_h(batch_size, rnn_ln, rnn_hdim))

        # hidden state could be used for single layer rnn
        if rnn.num_layers == 1:
            ht = ht[:, idx_unsort]

            if rnn.bidirectional:
                ht = torch.cat((ht[0], ht[1]), dim=1)
            else:
                ht = ht[0]

            return F.dropout(ht, p=rnn_out_dr, training=self.training)

        # unpack output
        # (L, B, rnn_hidden_size * num_directions)
        rnn_out, _ = pad_packed_sequence(rnn_out,
                                         batch_first=self.batch_first)

        # transpose
        # (B, L, rnn_hidden_size * num_directions)
        rnn_out = rnn_out.transpose(0, 1).contiguous()

        # unsort
        # rnn_out should be batch_first
        rnn_out = rnn_out[idx_unsort]
        tl = tl[idx_unsort]

        # flatten
        # (B * L, rnn_hidden_size * num_directions)
        rnn_out = rnn_out.view(-1, rnn_hdim * self.num_directions)
        if self.cuda_is_available:
            rnn_out = rnn_out.cuda()

        # select timestep by length
        fw_idxes = torch.arange(0, batch_size).type(
            torch.cuda.LongTensor if self.cuda_is_available
            else torch.LongTensor) * batch_max_seqlen + tl.data - 1

        selected_fw = rnn_out[fw_idxes]
        selected_fw = selected_fw[:, :rnn_hdim]

        # https://github.com/pytorch/pytorch/issues/3587#issuecomment-348340401
        # https://github.com/pytorch/pytorch/issues/3587#issuecomment-354284160
        if rnn.bidirectional:
            bw_idxes = torch.arange(0, batch_size).type(
                torch.cuda.LongTensor if self.cuda_is_available
                else torch.LongTensor) * batch_max_seqlen

            selected_bw = rnn_out[bw_idxes]
            selected_bw = selected_bw[:, rnn_hdim:]

            return F.dropout(torch.cat((selected_fw, selected_bw), 1),
                             p=rnn_out_dr,
                             training=self.training)
        else:
            return F.dropout(selected_fw,
                             p=rnn_out_dr,
                             training=self.training)

    @profile(__name__)
    def title_layer(self, tc, tw, tl, mode='t'):
        # it's snapshot size if mode='st'
        tl = Variable(torch.cuda.LongTensor(tl) if self.cuda_is_available
                      else torch.LongTensor(tl))
        batch_size = tl.size(0)
        batch_max_seqlen = tl.data.max()
        batch_max_wordlen = -1
        for tc_words in tc:
            for tc_word in tc_words:
                word_chars = len(tc_word)
                if batch_max_wordlen < word_chars:
                    batch_max_wordlen = word_chars
        assert batch_max_wordlen > -1

        # for tc_conv
        if batch_max_wordlen < self.tc_conv_min_dim:
            batch_max_wordlen = self.tc_conv_min_dim

        # assure char2idx[self.PAD] is 1 -> torch.ones()
        # (B, L, max_wordlen)
        tc_tensor = Variable(
            torch.ones((batch_size,
                        batch_max_seqlen,
                        batch_max_wordlen)).long())
        if self.cuda_is_available:
            tc_tensor = tc_tensor.cuda()
        for b_idx, (seq, seqlen) in enumerate(zip(tc, tl.data)):
            for w_idx in range(seqlen):
                word_chars = seq[w_idx]
                tc_tensor[b_idx, w_idx, :len(word_chars)] = \
                    torch.cuda.LongTensor(word_chars) \
                    if self.cuda_is_available \
                    else torch.LongTensor(word_chars)

        # assure word2idx[self.PAD] is 1 -> torch.ones()
        # (B, L)
        tw_tensor = Variable(
            torch.ones((batch_size, batch_max_seqlen)).long())
        if self.cuda_is_available:
            tw_tensor = tw_tensor.cuda()
        for idx, (seq, seqlen) in enumerate(zip(tw, tl.data)):
            tw_tensor[idx, :seqlen] = \
                torch.cuda.LongTensor(seq[:seqlen]) \
                if self.cuda_is_available \
                else torch.LongTensor(seq[:seqlen])

        # sort tc_tensor and tw_tensor by seq len
        tl, perm_idxes = tl.sort(dim=0, descending=True)
        tc_tensor = tc_tensor[perm_idxes]
        tw_tensor = tw_tensor[perm_idxes]

        # to be used after RNN to restore the order
        _, idx_unsort = torch.sort(perm_idxes, dim=0, descending=False)

        # character embedding for title character
        # (B * L, max_wordlen, char_embed_dim)
        tc_embed = self.char_embed(tc_tensor.view(-1, batch_max_wordlen))
        # tc_embed = Variable(torch.zeros(tc_embed.size()).cuda())
        if self.config.char_dr > 0:
            tc_embed = F.dropout(tc_embed,
                                 p=self.config.char_dr, training=self.training)

        # unsqueeze dim 2 and transpose
        # (B * L, char_embed_dim, 1, max_wordlen)
        tc_embed = torch.transpose(torch.unsqueeze(tc_embed, 2), 1, 3)

        # tc conv
        conv_result = list()
        for i, (conv, conv_bn) in enumerate(zip(self.tc_conv, self.tc_conv_bn)):
            tc_conv = torch.squeeze(conv(tc_embed), dim=2)
            tc_mp = torch.max(torch.tanh(conv_bn(tc_conv)), 2)[0]
            tc_mp = tc_mp.view(-1, batch_max_seqlen, tc_mp.size(1))
            conv_result.append(tc_mp)
        conv_result = torch.cat(conv_result, dim=2)

        # word embedding for title
        # (B, L, word_embed_dim)
        tw_embed = self.word_embed(tw_tensor)
        if self.config.word_dr > 0:
            tw_embed = F.dropout(tw_embed,
                                 p=self.config.word_dr,
                                 training=self.training)

        if not self.batch_first:
            # (L, B, sum(tc_conv_fn))
            conv_result = conv_result.transpose(0, 1)

            # (L, B, word_embed_dim)
            tw_embed = tw_embed.transpose(0, 1)

        # concat title character conv result and title word embedding
        # (L, B, sum(tc_conv_fn) + word_embed_dim)
        lstm_input = torch.cat((conv_result, tw_embed), 2)

        # pack, response for variable length batch
        packed_lstm_input = \
            pack_padded_sequence(lstm_input, tl.data.tolist(),
                                 batch_first=self.batch_first)

        # for input title
        if mode == 't':
            assert not self.config.no_title
            return self.get_rnn_out(batch_size, batch_max_seqlen, tl,
                                    packed_lstm_input, idx_unsort,
                                    self.t_rnn,
                                    self.config.t_rnn_hdim,
                                    self.config.t_rnn_out_dr,
                                    self.config.t_rnn_ln)
        # for snapshot title
        elif mode == 'st':
            assert not self.config.no_snapshot \
                   and not self.config.no_snapshot_title
            return self.get_rnn_out(batch_size, batch_max_seqlen, tl,
                                    packed_lstm_input, idx_unsort,
                                    self.st_rnn,
                                    self.config.st_rnn_hdim,
                                    self.config.st_rnn_out_dr,
                                    self.config.st_rnn_ln)
        else:
            raise ValueError('Invalid mode %s' % mode)

    @profile(__name__)
    def intention_layer(self, user, dur, title):
        # Highway network on concat
        if not self.config.no_title:
            concat = torch.cat((user, dur, title), 1)
        else:
            concat = torch.cat((user, dur), 1)
        nonl = F.rrelu(self.it_nonl(concat))
        gate = F.sigmoid(self.it_gate(concat))
        return torch.mul(gate, nonl) + torch.mul(1 - gate, concat)

    @profile(__name__)
    def snapshot_title_layer(self, stc, stw, stl):
        stacked_tc = []
        stacked_tw = []
        stacked_tl = []
        split_idx = [0]
        split_titles = []

        # Stack snapshot features
        for tc, tw, tl in zip(stc, stw, stl):
            stacked_tc += tc
            stacked_tw += tw
            stacked_tl += tl
            split_idx += [len(tc)]
        split_idx = np.cumsum(np.array(split_idx))

        # Run title layer once 
        if len(stacked_tc) > 0:
            snapshot_titles = self.title_layer(
                stacked_tc, stacked_tw, stacked_tl, mode='st')
        
        # Gather by split idx
        for s, e in zip(split_idx[:-1], split_idx[1:]):
            if s == e:
                split_titles.append(self.empty_st_rnn_output)
            else:
                split_titles.append(snapshot_titles[s:e])

        return split_titles

    @profile(__name__)
    def snapshot_layer(self, user_embed, stitle, sdur, sslot):
        # # test
        # return Variable(
        #     torch.zeros(user_embed.size(0), self.snapshot_odim).cuda())

        snapshot_rep_list = list()
        if not self.config.no_snapshot_title:
            for usr_emb, title, dur, slot \
                    in zip(user_embed, stitle, sdur, sslot):
                # if 0 == len(dur):
                #     snapshot_rep_list.append(
                #         Variable(torch.zeros(1, self.snapshot_odim).cuda()))
                # else:

                if 0 == len(dur):
                    dur = self.emtpy_long
                else:
                    dur = Variable(torch.cuda.LongTensor(dur)
                                   if self.cuda_is_available
                                   else torch.LongTensor(dur))

                if 0 == len(slot):
                    slot = self.emtpy_long
                else:
                    slot = Variable(torch.cuda.LongTensor(slot)
                                    if self.cuda_is_available
                                    else torch.LongTensor(slot))

                usr_emb = torch.unsqueeze(usr_emb, 0)
                snapshot_rep, _ = \
                    self.snapshot_layer_core(usr_emb, title, dur, slot)
                snapshot_rep_list.append(snapshot_rep)
        else:
            for usr_emb, dur, slot in zip(user_embed, sdur, sslot):
                # if 0 == len(dur):
                #     snapshot_rep_list.append(
                #         Variable(torch.zeros(1, self.snapshot_odim).cuda()))
                # else:

                if 0 == len(dur):
                    dur = self.emtpy_long
                else:
                    dur = Variable(torch.cuda.LongTensor(dur)
                                   if self.cuda_is_available
                                   else torch.LongTensor(dur))

                if 0 == len(slot):
                    slot = self.emtpy_long
                else:
                    slot = Variable(torch.cuda.LongTensor(slot)
                                    if self.cuda_is_available
                                    else torch.LongTensor(slot))

                usr_emb = torch.unsqueeze(usr_emb, 0)
                snapshot_rep, _ = \
                    self.snapshot_layer_core(usr_emb, None, dur, slot)
                snapshot_rep_list.append(snapshot_rep)
        return torch.cat(snapshot_rep_list, dim=0)

    @profile(__name__)
    def snapshot_layer_core(self, user_embed, title, dur, slot):
        new_slot = None
        snapshot_contents = None

        # ready for snapshot (contents)
        total_slots = self.config.sm_day_num * self.config.sm_slot_num
        saved_slot = list()
        if len(dur.size()) > 0:
            if not self.config.no_snapshot_title:
                assert title is not None
                title = title.data
                new_title = list()
                dur = dur / 30 - 1
                new_slot = list()

                assert title.size(0) == dur.size(0) == slot.size(0), \
                    't %d, d %d, s %d' % (
                    title.size(0), dur.size(0), slot.size(0))

                for i, (d, s) in enumerate(zip(dur.data, slot.data)):
                    new_slot.append(s)
                    new_title.append(title[i])
                    for k in range(d):
                        if s + k + 1 < total_slots:
                            new_slot.append(s + k + 1)
                            new_title.append(title[i])
                new_slot = np.array(new_slot)
                saved_slot = new_slot[:]
                new_slot = Variable(torch.cuda.LongTensor(new_slot)
                                    if self.cuda_is_available
                                    else torch.LongTensor(new_slot))
                new_title = \
                    torch.cat(new_title, 0).\
                    view(-1, self.config.st_rnn_hdim * self.num_directions)
                slot_embed = F.dropout(self.slot_embed(new_slot),
                                       p=self.config.slot_dr,
                                       training=self.training)
                slot_embed = slot_embed.view(-1, self.config.slot_embed_dim)
                # slot_embed = Variable(torch.zeros(slot_embed.size()).cuda())
                user_src_embed = user_embed.expand(slot_embed.size(0),
                                                   user_embed.size(1))
                snapshot_contents = \
                    torch.cat((new_title, user_src_embed.data, slot_embed.data),
                              1)
            else:
                dur = dur / 30 - 1
                new_slot = list()

                assert dur.size(0) == slot.size(0), \
                    'd %d, s %d' % (dur.size(0), slot.size(0))

                for i, (d, s) in enumerate(zip(dur.data, slot.data)):
                    new_slot.append(s)
                    for k in range(d):
                        if s + k + 1 < total_slots:
                            new_slot.append(s + k + 1)
                new_slot = np.array(new_slot)
                saved_slot = new_slot[:]
                new_slot = Variable(torch.cuda.LongTensor(new_slot)
                                    if self.cuda_is_available
                                    else torch.LongTensor(new_slot))
                slot_embed = F.dropout(self.slot_embed(new_slot),
                                       p=self.config.slot_dr,
                                       training=self.training)
                slot_embed = slot_embed.view(-1, self.config.slot_embed_dim)
                # slot_embed = Variable(torch.zeros(slot_embed.size()).cuda())
                user_src_embed = user_embed.expand(slot_embed.size(0),
                                                   user_embed.size(1))
                snapshot_contents = \
                    torch.cat((user_src_embed.data, slot_embed.data), 1)

        saved_slot = Variable(torch.cuda.LongTensor(saved_slot)
                              if self.cuda_is_available
                              else torch.LongTensor(saved_slot))

        # ready for slot, user embed (base)
        slot_all = Variable(torch.arange(0, total_slots).long())
        if self.cuda_is_available:
            slot_all = slot_all.cuda()
        slot_all_embed = self.slot_embed(slot_all)
        user_all_embed = user_embed[0].expand(slot_all_embed.size(0),
                                              user_embed.size(1))

        if not self.config.no_snapshot_title:
            zero_concat = \
                torch.zeros(
                    total_slots,
                    self.config.st_rnn_hdim * self.num_directions)
            if self.cuda_is_available:
                zero_concat = zero_concat.cuda()
            snapshot_base = torch.cat((zero_concat, user_all_embed.data,
                                       slot_all_embed.data), 1)
        else:
            snapshot_base = torch.cat((user_all_embed.data,
                                       slot_all_embed.data), 1)

        # ready for snapshot map (empty)
        snapshot_map = \
            Variable(torch.zeros(total_slots, self.sm_conv1_idim))
        if self.cuda_is_available:
            snapshot_map = snapshot_map.cuda()

        index = None
        if len(dur.size()) > 0:
            index = new_slot.data.unsqueeze(1)
            index = index.expand_as(snapshot_contents)
        slot_all = slot_all.data.unsqueeze(1)
        slot_all = slot_all.expand_as(snapshot_base)

        # scatter base and then the contents
        snapshot_map.data.scatter_(0, slot_all, snapshot_base)
        if len(dur.size()) > 0:
            snapshot_map.data.scatter_(0, index, snapshot_contents)

        # (sm_day_num, sm_slot_num,
        #  user_embed_dim + slot_embed_dim + st_rnn_hdim * num_directions)
        snapshot_map = snapshot_map.view(self.config.sm_day_num,
                                         self.config.sm_slot_num,
                                         self.sm_conv1_idim)

        # (user_embed_dim + slot_embed_dim + st_rnn_hdim * num_directions,
        #  sm_day_num, sm_slot_num)
        snapshot_map = torch.transpose(
            torch.transpose(snapshot_map, 0, 2), 1, 2)

        # multiple filter conv
        conv_list = [self.sm_conv1, self.sm_conv2]
        snapshot_mf = torch.unsqueeze(snapshot_map, 0)
        if self.cuda_is_available:
            snapshot_mf = snapshot_mf.cuda()

        for layer_idx, sm_conv in enumerate(conv_list):
            conv_result = list()
            for filter_idx, conv in enumerate(sm_conv):
                conv_out = conv(snapshot_mf)
                conv_result.append(conv_out)
            snapshot_mf = torch.cat(conv_result, 1)
            if layer_idx < len(conv_list) - 1:
                snapshot_mf = F.rrelu(self.sm_conv1_bn(snapshot_mf))
            else:
                snapshot_mf = torch.max(self.sm_conv2_bn(snapshot_mf.view(
                    1, snapshot_mf.size(1), -1)), 2)[0]

        return snapshot_mf, saved_slot

    @profile(__name__)
    def matching_layer(self, title, intention, snapshot_mf, grid):
        # Highway network for mf
        concat_seq = list()
        if not self.config.no_snapshot:
            if self.cuda_is_available:
                grid = grid.cuda()
            concat_seq.append(Variable(grid))
            concat_seq.insert(0, snapshot_mf)
        if not self.config.no_intention:
            concat_seq.insert(0, intention)
        else:
            if not self.config.no_title:
                concat_seq.insert(0, title)
        assert len(concat_seq) > 0

        if len(concat_seq) > 1:
            concat = torch.cat(concat_seq, 1)
        else:
            concat = concat_seq[0]

        nonl = F.rrelu(self.mt_nonl(concat))
        gate = F.sigmoid(self.mt_gate(concat))
        output = torch.mul(gate, nonl) + torch.mul(1 - gate, concat)
        output = F.dropout(output, p=self.config.output_dr,
                           training=self.training)

        return self.output_fc1(output)

    @profile(__name__)
    def forward(self, user, dur, tc, tw, tl, stc, stw, stl, sdur, sslot, gr):
        """
        11 Features
            - user: [batch]
            - dur: [batch]
            - tc: [batch, sentlen, wordlen]
            - tw: [batch, sentlen]
            - tl: [batch]
            - stc: [batch, snum, sentlen, wordlen]
            - stw: [batch, snum, sentlen]
            - stl: [batch, snum]
            - sdur: [batch, snum]
            - sslot: [batch, snum]
            - gr: [batch, snum]
        """

        title_rep = None
        if not self.config.no_title:
            # (B, t_rnn_hdim * num_directions)
            title_rep = self.title_layer(tc, tw, tl)

        user_embed = None
        if not self.config.no_intention or not self.config.no_snapshot:
            if self.cuda_is_available:
                user = user.cuda()

            user_embed = self.user_embed(Variable(user))
            # user_embed = Variable(torch.zeros(user_embed.size()))

            if self.config.user_dr > 0:
                user_embed = F.dropout(user_embed,
                                       p=self.config.user_dr,
                                       training=self.training)

        intention_rep = None
        if not self.config.no_intention:
            if self.cuda_is_available:
                dur = dur.cuda()

            dur_embed = self.dur_embed(Variable(dur))
            # dur_embed = Variable(torch.zeros(dur_embed.size()))

            if self.config.dur_dr > 0:
                dur_embed = F.dropout(dur_embed,
                                      p=self.config.dur_dr,
                                      training=self.training)

            # (B, user_embed_dim + dur_embed_dim + t_rnn_hdim * num_directions)
            intention_rep = \
                self.intention_layer(user_embed, dur_embed, title_rep)

        if not self.config.no_snapshot:
            stitle_rep = None
            if not self.config.no_snapshot_title:
                # (B, (VARIABLE snapshot length, st_rnn_hdim * num_directions))
                stitle_rep = self.snapshot_title_layer(stc, stw, stl)

            # (B, sum(config.sm_conv_fn[len(config.sm_conv_fn)//2:]))
            snapshot_mf = \
                self.snapshot_layer(user_embed, stitle_rep, sdur, sslot)

            # (B, config.sm_day_num * config.sm_slot_num)
            output = \
                self.matching_layer(title_rep, intention_rep, snapshot_mf, gr)
        else:
            output = self.matching_layer(title_rep, intention_rep, None, None)

        assert output.size(1) == \
            self.config.sm_day_num * self.config.sm_slot_num
        return output, (title_rep, intention_rep)

    def get_regloss(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.config.wd
        reg_loss = 0
        params = [self.output_fc1, self.output_fc2, self.it_nonl, self.it_gate]
        for param in params:
            reg_loss += torch.norm(param.weight, 2)
        return reg_loss * weight_decay

    def decay_lr(self, lr_decay=None):
        if lr_decay is None:
            lr_decay = self.config.lr_decay
        self.config.lr /= lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr

        print('\tlearning rate decay to %.3f' % self.config.lr)

    @profile(__name__)
    def get_metrics(self, outputs, targets, ex_targets=None):
        outputs_max_idxes = torch.squeeze(torch.topk(outputs, 1)[1], dim=1).data
        outputs_topall_idxes = torch.topk(outputs, self.n_classes)[1].data
        targets = targets.data.cpu()
        outputs = outputs.data.cpu()

        topk = 10
        ex1 = 0.
        ex5 = 0.
        ex10 = 0.
        for o, et, t in zip(outputs, ex_targets, targets):
            assert et[t] == 0, et[t]
            output = o[:] - (et[:] * 1e16)
            output_topk = torch.topk(output, topk)[1]

            if t == output_topk[0]:
                ex1 += 1.
            if t in output_topk[:5]:
                ex5 += 1.
            if t in output_topk:
                ex10 += 1.

        def ndcg_at_k(r, k):
            def get_dcg(_r, _k):
                _dcg = 0.
                for rk_idx, rk in enumerate(_r):
                    if rk_idx == _k:
                        break
                    _dcg += ((2 ** rk) - 1) / math.log2(2 + rk_idx)
                return _dcg

            return get_dcg(r, k) / get_dcg(sorted(r, reverse=True), k)

        def inverse_euclidean_distance(target, pred):
            euc = (((pred // self.n_day_slots) - (target // self.n_day_slots))
                   ** 2
                   + ((pred % self.n_day_slots) - (target % self.n_day_slots))
                   ** 2) ** 0.5
            return 1. / (euc + 1.)

        mrr = 0.
        ndcg_at_5 = 0.
        ndcg_at_10 = 0.
        for target_slot_idx, ota in zip(targets, outputs_topall_idxes):
            # relevance vector for nDCG
            relevance_vector = [0.] * self.n_classes

            target_rank_idx = -1
            for rank_idx, slot_idx in enumerate(ota):
                if target_slot_idx == slot_idx:
                    target_rank_idx = rank_idx

                # assign ieuc
                relevance_vector[rank_idx] = \
                    inverse_euclidean_distance(target_slot_idx, slot_idx)

            assert target_rank_idx > -1

            # MRR
            mrr += 1. / (target_rank_idx + 1.)

            # nDCG@5 and nDCG@10
            ndcg_at_5 += ndcg_at_k(relevance_vector, 5)
            ndcg_at_10 += ndcg_at_k(relevance_vector, 10)

        ieuc = 0.
        for t, m in zip(targets, outputs_max_idxes):
            ieuc += inverse_euclidean_distance(t, m)

        len_outputs = len(outputs)
        ex1 /= len_outputs
        ex5 /= len_outputs
        ex10 /= len_outputs
        mrr /= len_outputs
        ieuc /= len_outputs
        ndcg_at_5 /= len_outputs
        ndcg_at_10 /= len_outputs

        return ex1, ex5, ex10, mrr, ieuc, ndcg_at_5, ndcg_at_10

    def save_checkpoint(self, state, filename=None):
        if filename is None:
            filename = os.path.join(self.config.checkpoint_dir,
                                    self.config.model_name + '.pth')
        else:
            filename = os.path.join(self.config.checkpoint_dir,
                                    filename + '.pth')
        print('\t=> save checkpoint %s' % filename)
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None, map_location=None):
        if filename is None:
            filename = self.config.checkpoint_dir + self.config.model_name
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> load checkpoint %s' % filename)
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.config = checkpoint['config']

    @profile(__name__)
    def write_summary(self, mode, metrics, offset):
        if mode == 'tr':
            writer = self.train_writer
        elif mode == 'va':
            writer = self.valid_writer
        elif mode == 'te':
            writer = self.test_writer
        else:
            raise ValueError('Invalid mode %s' % mode)

        writer.add_scalar('loss', metrics[0], offset)
        writer.add_scalar('recall@1', metrics[1], offset)
        writer.add_scalar('recall@5', metrics[2], offset)
        writer.add_scalar('recall@10', metrics[3], offset)
        writer.add_scalar('mrr', metrics[4], offset)
        writer.add_scalar('ieuc', metrics[5], offset)
        writer.add_scalar('ndcg@5', metrics[6], offset)
        writer.add_scalar('ndcg@10', metrics[7], offset)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            writer.add_histogram(name, param.clone().cpu().data.numpy(), offset)

    def close_summary_writer(self):
        self.train_writer.close()
        self.valid_writer.close()
        self.test_writer.close()
