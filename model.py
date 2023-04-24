import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import os
from torch.utils.tensorboard import SummaryWriter
from utils import Profile


class NESA(nn.Module):
    def __init__(self, config, widx2vec, idx2dur=None, class_weight=None,
                 idx=None):
        super(NESA, self).__init__()
        self.config = config

        use_cuda = self.config.yes_cuda > 0 and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.n_classes = config.slot_size // config.class_div
        self.n_day_slots = self.n_classes // 7

        # embedding layers
        self.char_embed = nn.Embedding(config.char_vocab_size,
                                       config.char_embed_dim,
                                       padding_idx=0)
        self.word_embed = nn.Embedding(config.word_vocab_size,
                                       config.word_embed_dim,
                                       padding_idx=0)

        if not config.no_intention or not config.no_context:
            self.user_embed = nn.Embedding(config.user_size,
                                           config.user_embed_dim)
        if not config.no_intention:
            self.dur_embed = nn.Embedding(config.dur_size, config.dur_embed_dim)
            if config.use_duration_scala > 0:
                assert idx2dur is not None
                self.dur_embed = nn.Embedding(config.dur_size, 1)
                didx2vec = np.zeros((config.dur_size, 1))
                for dur_idx in idx2dur:
                    # zero to one
                    didx2vec[dur_idx] = min(1., idx2dur[dur_idx] / 720.)
                self.dur_embed.weight.data.copy_(torch.from_numpy(didx2vec))
                self.dur_embed.weight.requires_grad = False

        if not config.no_context:
            self.slot_embed = nn.Embedding(self.n_classes,
                                           config.slot_embed_dim)
            self.emtpy_long = torch.LongTensor([]).to(self.device)

        # dimensions according to settings
        self.num_directions = config.num_directions
        self.t_rnn_idim = config.word_embed_dim + sum(config.tc_conv_fn)
        self.st_rnn_idim = config.word_embed_dim + sum(config.tc_conv_fn)
        self.sm_conv1_idim = config.user_embed_dim + config.slot_embed_dim
        if not config.no_context and not config.no_context_title:
            self.sm_conv1_idim += config.st_rnn_hdim * self.num_directions
            self.empty_st_rnn_output = \
                torch.zeros(1, self.config.st_rnn_hdim * self.num_directions) \
                .to(self.device)
        self.sm_conv2_idim = sum(
            config.sm_conv_fn[:len(config.sm_conv_fn) // 2])

        self.it_idim = config.user_embed_dim + config.dur_embed_dim
        if not config.no_title:
            self.it_idim += config.t_rnn_hdim * self.num_directions
        self.mt_idim = 0
        if not config.no_intention:
            self.mt_idim += self.it_idim
        else:
            if not config.no_title:
                self.mt_idim += config.t_rnn_hdim * self.num_directions
        if not config.no_context:
            self.context_odim = sum(
                config.sm_conv_fn[len(config.sm_conv_fn) // 2:])

            self.mt_idim += config.sm_day_num * config.sm_slot_num
            self.mt_idim += self.context_odim

        # convolution layers
        self.tc_conv = nn.ModuleList(
            [nn.Conv2d(config.char_embed_dim, config.tc_conv_fn[i],
                       (config.tc_conv_fh[i], config.tc_conv_fw[i]),
                       stride=1) for i in range(len(config.tc_conv_fn))])
        self.tc_conv_bn = nn.ModuleList(
            [nn.BatchNorm2d(num_tc_conv_f)
             for num_tc_conv_f in config.tc_conv_fn])
        self.tc_conv_min_dim = len(config.tc_conv_fn) + 1

        if not config.no_context:
            self.sm_conv1 = nn.ModuleList([nn.Conv2d(
                self.sm_conv1_idim, config.sm_conv_fn[i],
                (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                stride=1, padding=config.sm_conv_pd[i])
                for i in range(0, len(config.sm_conv_fn) // 2)])
            self.sm_mp1 = nn.MaxPool2d(2)
            self.sm_conv1_bn = nn.BatchNorm2d(self.sm_conv2_idim)
            self.sm_conv2 = nn.ModuleList([nn.Conv2d(
                self.sm_conv2_idim,
                config.sm_conv_fn[i + len(config.sm_conv_fn) // 2],
                (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                stride=1, padding=config.sm_conv_pd[i])
                for i in range(len(config.sm_conv_fn) // 2)])
            self.sm_mp2 = nn.MaxPool2d(2)
            self.sm_conv2_bn = nn.BatchNorm2d(self.context_odim)

        # rnn layers
        self.batch_first = False
        self.bidirectional = config.num_directions == 2

        if not config.no_title:
            self.t_rnn = nn.LSTM(self.t_rnn_idim, config.t_rnn_hdim,
                                 config.t_rnn_ln,
                                 dropout=config.t_rnn_dr,
                                 batch_first=self.batch_first,
                                 bidirectional=self.bidirectional)

        if not config.no_context and not config.no_context_title:
            self.st_rnn = nn.LSTM(self.st_rnn_idim, config.st_rnn_hdim,
                                  config.st_rnn_ln,
                                  dropout=config.st_rnn_dr,
                                  batch_first=self.batch_first,
                                  bidirectional=self.bidirectional)

        # linear layers
        if not config.no_intention:
            self.it_nonl = nn.Linear(self.it_idim, self.it_idim)
            self.it_gate = nn.Linear(self.it_idim, self.it_idim)
        self.mt_nonl = nn.Linear(self.mt_idim, self.mt_idim)
        self.mt_gate = nn.Linear(self.mt_idim, self.mt_idim)
        self.output_fc1 = nn.Linear(self.mt_idim,
                                    config.sm_day_num * config.sm_slot_num)

        # initialization
        self.init_word_embed(widx2vec,
                             requires_grad=config.word_embed_req_grad > 0)
        self.init_convs()
        self.init_linears()

        self.params = self.model_params(debug=False)
        self.optimizer = optim.Adam(self.params, lr=config.lr,
                                    weight_decay=config.wd,
                                    amsgrad=True)
        self.scheduler = \
            optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                 factor=0.5,
                                                 patience=1)

        # https://discuss.pytorch.org/t/loss-weighting-imbalanced-data/11698
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        if config.summary:
            summary_path = 'runs/' + config.model_name + \
                           ('_%d' % idx if idx is not None else '')
            self.summary_writer = SummaryWriter(log_dir=summary_path)

    def init_word_embed(self, widx2vec, requires_grad=False):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(widx2vec)))
        self.word_embed.weight.requires_grad = requires_grad

    def init_convs(self):
        def init_conv_list(conv_list):
            for conv in conv_list:
                # https://discuss.pytorch.org/t/weight-initilzation/157/9
                nn.init.xavier_uniform_(conv.weight.data)
                nn.init.uniform_(conv.bias.data)

        init_conv_list(self.tc_conv)

        if not self.config.no_context:
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
                nn.init.xavier_uniform_(self.it_nonl.weight,
                                        gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(self.it_nonl.bias)
                nn.init.xavier_uniform_(self.it_gate.weight, gain=1)
                nn.init.uniform_(self.it_gate.bias)
            nn.init.xavier_uniform_(self.mt_nonl.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.uniform_(self.mt_nonl.bias)
            nn.init.xavier_uniform_(self.mt_gate.weight, gain=1)
            nn.init.uniform_(self.mt_gate.bias)
            nn.init.xavier_uniform_(self.output_fc1.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.uniform_(self.output_fc1.bias)
        elif 'uniform' == init:
            stdv_pow = 0.5
            if not self.config.no_intention:
                linear_init_uniform(self.it_nonl, stdv_power=stdv_pow)
                linear_init_uniform(self.it_gate, stdv_power=stdv_pow)
            linear_init_uniform(self.mt_nonl, stdv_power=stdv_pow)
            linear_init_uniform(self.mt_gate, stdv_power=stdv_pow)
            linear_init_uniform(self.output_fc1, stdv_power=stdv_pow)

    def init_rnn_h(self, batch_size, rnn_ln, hdim):
        h_0 = torch.zeros(rnn_ln * self.num_directions,
                          batch_size,
                          hdim).to(self.device)
        c_0 = torch.zeros(rnn_ln * self.num_directions,
                          batch_size,
                          hdim).to(self.device)
        return h_0, c_0

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
                    packed_input, idx_unsort,
                    rnn, rnn_hdim, rnn_out_dr, rnn_ln):
        assert idx_unsort is not None

        # rnn
        rnn_out, (ht, ct) = rnn(packed_input,
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
        rnn_out = \
            rnn_out.view(-1, rnn_hdim * self.num_directions).to(self.device)

        # select timestep by length
        fw_idxes = \
            torch.arange(0, batch_size, dtype=torch.long) \
            .to(self.device) * batch_max_seqlen + tl - 1

        selected_fw = rnn_out[fw_idxes]
        selected_fw = selected_fw[:, :rnn_hdim]

        # https://github.com/pytorch/pytorch/issues/3587#issuecomment-348340401
        # https://github.com/pytorch/pytorch/issues/3587#issuecomment-354284160
        if rnn.bidirectional:
            bw_idxes = \
                torch.arange(0, batch_size, dtype=torch.long) \
                .to(self.device) * batch_max_seqlen

            selected_bw = rnn_out[bw_idxes]
            selected_bw = selected_bw[:, rnn_hdim:]

            return F.dropout(torch.cat((selected_fw, selected_bw), 1),
                             p=rnn_out_dr,
                             training=self.training)
        else:
            return F.dropout(selected_fw,
                             p=rnn_out_dr,
                             training=self.training)

    @Profile(__name__)
    def title_layer(self, tc, tw, tl, mode='t'):
        # it's context size if mode='st'
        tl = torch.LongTensor(tl).to(self.device)
        batch_size = tl.size(0)  # B
        batch_max_seqlen = tl.max()  # L
        batch_max_wordlen = -1
        for tc_words in tc:
            for tc_word in tc_words:
                word_chars = len(tc_word)
                if batch_max_wordlen < word_chars:
                    batch_max_wordlen = word_chars
        assert batch_max_wordlen > -1

        # force padding for tc_conv
        if batch_max_wordlen < self.tc_conv_min_dim:
            batch_max_wordlen = self.tc_conv_min_dim

        # assure that dataset.char2idx[self.PAD] is 0
        # (B, L (batch_max_seqlen), max_wordlen)
        tc_tensor = torch.zeros((batch_size,
                                 batch_max_seqlen,
                                 batch_max_wordlen), dtype=torch.long) \
            .to(self.device)
        for b_idx, (seq, seqlen) in enumerate(zip(tc, tl)):
            for w_idx in range(seqlen):
                word_chars = seq[w_idx]
                tc_tensor[b_idx, w_idx, :len(word_chars)] = \
                    torch.LongTensor(word_chars).to(self.device)

        # assure that dataset.word2idx[self.PAD] is 0
        # (B, L (batch_max_seqlen))
        tw_tensor = torch.zeros((batch_size,
                                 batch_max_seqlen), dtype=torch.long) \
            .to(self.device)
        for idx, (seq, seqlen) in enumerate(zip(tw, tl)):
            tw_tensor[idx, :seqlen] = \
                torch.LongTensor(seq[:seqlen]).to(self.device)

        # sort tc_tensor and tw_tensor by seq len
        tl, perm_idxes = tl.sort(dim=0, descending=True)
        tc_tensor = tc_tensor[perm_idxes]
        tw_tensor = tw_tensor[perm_idxes]

        # to be used after RNN to restore the order
        _, idx_unsort = torch.sort(perm_idxes, dim=0, descending=False)

        # character embedding for title character
        # (B * L (batch_max_seqlen), max_wordlen, char_embed_dim)
        tc_embed = self.char_embed(tc_tensor.view(-1, batch_max_wordlen))
        # tc_embed = torch.zeros(tc_embed.size()).to(self.device)
        if self.config.char_dr > 0:
            tc_embed = F.dropout(tc_embed,
                                 p=self.config.char_dr, training=self.training)

        # unsqueeze dim 2 and transpose
        # (B * L (batch_max_seqlen), char_embed_dim, 1, max_wordlen)
        tc_embed = torch.transpose(torch.unsqueeze(tc_embed, 2), 1, 3)

        # tc conv
        # (N, channels, height, width)
        conv_result = list()
        for i, (conv, conv_bn) in enumerate(zip(self.tc_conv, self.tc_conv_bn)):
            tc_conv = conv(tc_embed)

            tc_mp = torch.max(torch.tanh(conv_bn(tc_conv)), 3)[0]

            # (B, L, tc_conv_fn[i])
            tc_mp = tc_mp.view(-1, batch_max_seqlen, tc_mp.size(1))

            conv_result.append(tc_mp)

        # (B, L, sum(tc_conv_fn))
        conv_result = torch.cat(conv_result, dim=2)

        # word embedding for title
        # (B, L, word_embed_dim)
        tw_embed = self.word_embed(tw_tensor)
        # # ablation: word embedding
        # tw_embed = torch.zeros(tw_embed.size()).to(self.device)

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
        rnn_input = torch.cat((conv_result, tw_embed), 2)

        # pack, response for variable length batch
        packed_input = \
            pack_padded_sequence(rnn_input, tl, batch_first=self.batch_first)

        # for input title
        if mode == 't':
            assert not self.config.no_title
            return self.get_rnn_out(batch_size, batch_max_seqlen, tl,
                                    packed_input, idx_unsort,
                                    self.t_rnn,
                                    self.config.t_rnn_hdim,
                                    self.config.t_rnn_out_dr,
                                    self.config.t_rnn_ln)
        # for context title
        elif mode == 'st':
            assert not self.config.no_context \
                   and not self.config.no_context_title
            return self.get_rnn_out(batch_size, batch_max_seqlen, tl,
                                    packed_input, idx_unsort,
                                    self.st_rnn,
                                    self.config.st_rnn_hdim,
                                    self.config.st_rnn_out_dr,
                                    self.config.st_rnn_ln)
        else:
            raise ValueError('Invalid mode %s' % mode)

    @Profile(__name__)
    def intention_layer(self, user, dur, title):
        # Highway network on concat
        if not self.config.no_title:
            concat = torch.cat((user, dur, title), 1)
        else:
            concat = torch.cat((user, dur), 1)
        nonl = F.rrelu(self.it_nonl(concat))
        gate = torch.sigmoid(self.it_gate(concat))
        return torch.mul(gate, nonl) + torch.mul(1 - gate, concat)

    @Profile(__name__)
    def context_title_layer(self, stc, stw, stl):
        stacked_tc = []
        stacked_tw = []
        stacked_tl = []
        split_idx = [0]
        split_titles = []

        # Stack context features
        for tc, tw, tl in zip(stc, stw, stl):
            stacked_tc += tc
            stacked_tw += tw
            stacked_tl += tl
            split_idx += [len(tc)]
        split_idx = np.cumsum(np.array(split_idx))

        # Run title layer once
        if len(stacked_tc) > 0:
            context_titles = self.title_layer(
                stacked_tc, stacked_tw, stacked_tl, mode='st')
        else:
            context_titles = self.empty_st_rnn_output

        # Gather by split idx
        for s, e in zip(split_idx[:-1], split_idx[1:]):
            if s == e:
                split_titles.append(self.empty_st_rnn_output)
            else:
                split_titles.append(context_titles[s:e])

        return split_titles

    @Profile(__name__)
    def context_layer(self, user_embed, stitle, sdur, sslot):
        # # test
        # return torch.zeros(user_embed.size(0), self.context_odim) \
        #     .to(self.device)

        context_rep_list = list()
        if not self.config.no_context_title:
            for usr_emb, title, dur, slot \
                    in zip(user_embed, stitle, sdur, sslot):
                # if 0 == len(dur):
                #     context_rep_list.append(
                #         torch.zeros(1, self.context_odim).to(self.device))
                # else:

                if 0 == len(dur):
                    dur = self.emtpy_long
                else:
                    dur = torch.LongTensor(dur).to(self.device)

                if 0 == len(slot):
                    slot = self.emtpy_long
                else:
                    slot = torch.LongTensor(slot).to(self.device)

                usr_emb = torch.unsqueeze(usr_emb, 0)
                context_rep, _ = \
                    self.context_layer_core(usr_emb, title, dur, slot)
                context_rep_list.append(context_rep)
        else:
            for usr_emb, dur, slot in zip(user_embed, sdur, sslot):
                # if 0 == len(dur):
                #     context_rep_list.append(
                #         torch.zeros(1, self.context_odim).to(self.device))
                # else:

                if 0 == len(dur):
                    dur = self.emtpy_long
                else:
                    dur = torch.LongTensor(dur).to(self.device)

                if 0 == len(slot):
                    slot = self.emtpy_long
                else:
                    slot = torch.LongTensor(slot).to(self.device)

                usr_emb = torch.unsqueeze(usr_emb, 0)
                context_rep, _ = \
                    self.context_layer_core(usr_emb, None, dur, slot)
                context_rep_list.append(context_rep)
        return torch.cat(context_rep_list, dim=0)

    @Profile(__name__)
    def context_layer_core(self, user_embed, title, dur, slot):
        new_slot = None
        context_contents = None

        # ready for context (contents)
        total_slots = self.config.sm_day_num * self.config.sm_slot_num
        saved_slot = list()

        has_preregistered_events = dur.size(0) > 0

        if has_preregistered_events:
            dur = torch.ceil(dur.float() / (30 * self.config.class_div)) \
                      .long() - 1
            new_slot = list()

            assert dur.size(0) == slot.size(0), \
                'd %d, s %d' % (dur.size(0), slot.size(0))

            if not self.config.no_context_title:
                assert title is not None
                new_title = list()

                assert title.size(0) == dur.size(0), \
                    't %d, d %d' % (title.size(0), dur.size(0))

                for i, (d, s) in enumerate(zip(dur, slot)):
                    if d < 0:
                        d = 0
                    new_slot.append(s)
                    new_title.append(title[i])
                    for k in range(d):
                        if s + k + 1 < total_slots:
                            new_slot.append(s + k + 1)
                            new_title.append(title[i])
                new_slot = np.array(new_slot)
                saved_slot = new_slot[:]
                new_slot = torch.LongTensor(new_slot).to(self.device)
                new_title = \
                    torch.cat(new_title, 0). \
                    view(-1, self.config.st_rnn_hdim * self.num_directions)
                slot_embed = F.dropout(self.slot_embed(new_slot),
                                       p=self.config.slot_dr,
                                       training=self.training)
                slot_embed = slot_embed.view(-1, self.config.slot_embed_dim)
                # slot_embed = torch.zeros(slot_embed.size()).to(self.device)
                user_src_embed = user_embed.expand(slot_embed.size(0),
                                                   user_embed.size(1))
                context_contents = \
                    torch.cat((new_title, user_src_embed, slot_embed), 1)
            else:
                for i, (d, s) in enumerate(zip(dur, slot)):
                    new_slot.append(s)
                    for k in range(d):
                        if d < 0:
                            d = 0
                        if s + k + 1 < total_slots:
                            new_slot.append(s + k + 1)
                new_slot = np.array(new_slot)
                saved_slot = new_slot[:]
                new_slot = torch.LongTensor(new_slot).to(self.device)
                slot_embed = F.dropout(self.slot_embed(new_slot),
                                       p=self.config.slot_dr,
                                       training=self.training)
                slot_embed = slot_embed.view(-1, self.config.slot_embed_dim)
                # slot_embed = torch.zeros(slot_embed.size()).to(self.device)
                user_src_embed = user_embed.expand(slot_embed.size(0),
                                                   user_embed.size(1))
                context_contents = torch.cat((user_src_embed, slot_embed), 1)

        saved_slot = torch.LongTensor(saved_slot).to(self.device)

        # ready for slot, user embed (base)
        slot_all = torch.arange(0, total_slots, dtype=torch.long) \
            .to(self.device)
        slot_all_embed = self.slot_embed(slot_all)
        user_all_embed = user_embed[0].expand(slot_all_embed.size(0),
                                              user_embed.size(1))

        if not self.config.no_context_title:
            zero_concat = \
                torch.zeros(
                    total_slots,
                    self.config.st_rnn_hdim * self.num_directions) \
                .to(self.device)
            context_base = torch.cat((zero_concat, user_all_embed,
                                      slot_all_embed), 1)
        else:
            context_base = torch.cat((user_all_embed, slot_all_embed), 1)

        # ready for context map (empty)
        context_map = torch.zeros(total_slots, self.sm_conv1_idim) \
            .to(self.device)

        index = None
        if has_preregistered_events:
            index = new_slot.unsqueeze(1)
            index = index.expand_as(context_contents)
        slot_all = slot_all.unsqueeze(1)
        slot_all = slot_all.expand_as(context_base)

        # scatter base and then the contents
        context_map.scatter_(0, slot_all, context_base)
        if has_preregistered_events:
            context_map.scatter_(0, index, context_contents)

        # (sm_day_num, sm_slot_num,
        #  user_embed_dim + slot_embed_dim + st_rnn_hdim * num_directions)
        context_map = context_map.view(self.config.sm_day_num,
                                       self.config.sm_slot_num,
                                       self.sm_conv1_idim)

        # (user_embed_dim + slot_embed_dim + st_rnn_hdim * num_directions,
        #  sm_day_num,
        #  sm_slot_num)
        context_map = context_map.permute(2, 0, 1)

        # multiple filter conv
        conv_list = [self.sm_conv1, self.sm_conv2]
        context_mf = torch.unsqueeze(context_map, 0).to(self.device)

        for layer_idx, sm_conv in enumerate(conv_list):
            conv_result = list()
            for filter_idx, conv in enumerate(sm_conv):
                conv_out = conv(context_mf)
                conv_result.append(conv_out)
            context_mf = torch.cat(conv_result, 1)
            if layer_idx == 0:
                context_mf = F.rrelu(self.sm_conv1_bn(context_mf))
            else:  # layer_idx == 1
                context_mf = torch.max(self.sm_conv2_bn(context_mf)
                                       .view(1, context_mf.size(1), -1), 2)[0]

        return context_mf, saved_slot

    @Profile(__name__)
    def matching_layer(self, title, intention, context_mf, grid):
        # Highway network for mf
        concat_seq = list()
        if not self.config.no_context:
            concat_seq.append(grid.to(self.device))
            concat_seq.insert(0, context_mf)
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
        gate = torch.sigmoid(self.mt_gate(concat))
        output = torch.mul(gate, nonl) + torch.mul(1 - gate, concat)
        output = F.dropout(output, p=self.config.output_dr,
                           training=self.training)

        return self.output_fc1(output)

    @Profile(__name__)
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
        if not self.config.no_intention or not self.config.no_context:
            user_embed = self.user_embed(user.to(self.device))
            # user_embed = torch.zeros(user_embed.size()).to(self.device)

            if self.config.user_dr > 0:
                user_embed = F.dropout(user_embed,
                                       p=self.config.user_dr,
                                       training=self.training)

        intention_rep = None
        if not self.config.no_intention:
            dur_embed = self.dur_embed(dur.to(self.device))
            # dur_embed = torch.zeros(dur.size(0), self.config.dur_embed_dim) \
            #     .to(self.device)

            if self.config.dur_dr > 0:
                dur_embed = F.dropout(dur_embed,
                                      p=self.config.dur_dr,
                                      training=self.training)

            # (B, user_embed_dim + dur_embed_dim + t_rnn_hdim * num_directions)
            intention_rep = \
                self.intention_layer(user_embed, dur_embed, title_rep)

        if not self.config.no_context:
            stitle_rep = None
            if not self.config.no_context_title:
                # (B, (VARIABLE context length, st_rnn_hdim * num_directions))
                stitle_rep = self.context_title_layer(stc, stw, stl)

            # (B, sum(config.sm_conv_fn[len(config.sm_conv_fn)//2:]))
            context_mf = self.context_layer(user_embed, stitle_rep, sdur, sslot)

            # (B, config.sm_day_num * config.sm_slot_num)
            output = \
                self.matching_layer(title_rep, intention_rep, context_mf, gr)
        else:
            output = self.matching_layer(title_rep, intention_rep, None, None)

        assert output.size(1) == \
            self.config.sm_day_num * self.config.sm_slot_num
        return output

    def get_regloss(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.config.wd
        reg_loss = 0
        params = [self.output_fc1, self.it_nonl, self.it_gate]
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

    def save_checkpoint(self, state, filename=None):
        if filename is None:
            filename = os.path.join(self.config.checkpoint_dir,
                                    self.config.model_name + '.pth')
        else:
            filename = os.path.join(self.config.checkpoint_dir,
                                    filename + '.pth')
        print('\t-> save checkpoint %s' % filename)
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = os.path.join(self.config.checkpoint_dir,
                                    self.config.model_name + '.pth')
        else:
            filename = os.path.join(self.config.checkpoint_dir,
                                    filename + '.pth')
        print('\t-> load checkpoint %s' % filename)
        checkpoint = torch.load(filename,
                                map_location=None if 'cuda' == self.device.type
                                else 'cpu')
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    @Profile(__name__)
    def write_summary(self, mode, loss, metrics, offset, add_histogram=False):
        if mode != 'tr':
            return

        self.summary_writer.add_scalar('loss', loss, offset)
        self.summary_writer.add_scalar('mrr', metrics[2], offset)

        if add_histogram:
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # add_histogram() takes lots of time
                self.summary_writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), offset)

    def close_summary_writer(self):
        self.summary_writer.close()


@Profile(__name__)
def get_metrics(outputs, targets, n_day_slots, n_classes, ex_targets=None,
                topk=5):
    if ex_targets is not None:
        for output, target, et in zip(outputs, targets, ex_targets):
            assert et[target] == 0
            output -= et * 99999.

    def get_recalls():
        def get_r1_r5(_o, _t):
            out_topk = torch.topk(_o, topk)[1]
            if _t == out_topk[0]:
                return 1., 1.
            else:
                if _t in out_topk:
                    return 0., 1.
            return 0., 0.

        ex1 = 0.
        ex5 = 0.
        for o, t in zip(outputs, targets):
            r1, r5 = get_r1_r5(o, t)
            ex1 += r1
            ex5 += r5
        return ex1, ex5

    def ndcg_at_k(r, k):
        def get_dcg(_r, _k):
            _dcg = 0.
            for rk_idx, rk in enumerate(_r):
                if rk_idx == _k:
                    break
                _dcg += ((2 ** rk) - 1) / math.log2(2 + rk_idx)
            return _dcg

        return get_dcg(r, k) / get_dcg(sorted(r, reverse=True), k)

    def inverse_euclidean_distance(_target, pred):
        euc = (((pred // n_day_slots) - (_target // n_day_slots))
               ** 2
               + ((pred % n_day_slots) - (_target % n_day_slots))
               ** 2) ** 0.5
        return 1. / (euc + 1.)

    def get_mrr_ndcg(calc_ndcg=False):
        mrr_sum = 0.
        ndcg_at_5_sum = 0.
        outputs_topall_idxes = torch.topk(outputs, n_classes)[1]

        # relevance vector for nDCG
        relevance_vector = [0.] * n_classes if calc_ndcg else None

        for target_slot_idx, ota in zip(targets, outputs_topall_idxes):
            target_rank_idx = -1
            for rank_idx, slot_idx in enumerate(ota):
                if -1 == target_rank_idx and target_slot_idx.item() == slot_idx.item():
                    target_rank_idx = rank_idx
                    if not calc_ndcg:
                        break

                if calc_ndcg:
                    # assign ieuc
                    relevance_vector[rank_idx] = \
                        inverse_euclidean_distance(slot_idx.item(),
                                                   target_slot_idx.item())

            assert target_rank_idx > -1

            # MRR
            mrr_sum += 1. / (target_rank_idx + 1)

            if calc_ndcg:
                # nDCG@5
                ndcg_at_5_sum += ndcg_at_k(relevance_vector, 5)
        return mrr_sum, ndcg_at_5_sum

    def get_ieuc():
        ieuc_sum = 0.
        outputs_max_idxes = torch.max(outputs, 1)[1]
        for m, t in zip(outputs_max_idxes, targets):
            ieuc_sum += inverse_euclidean_distance(t.item(), m.item())
        return ieuc_sum

    recall1, recall5 = get_recalls()
    mrr, _ = get_mrr_ndcg(calc_ndcg=False)
    ieuc = get_ieuc()

    return recall1, recall5, mrr, ieuc
