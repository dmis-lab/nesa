import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
# import sys
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class NETS(nn.Module):
    def __init__(self, config, widx2vec, iter=None):
        super(NETS, self).__init__()
        self.config = config

        # embedding layers
        self.char_embed = nn.Embedding(config.char_vocab_size, config.char_embed_dim,
                padding_idx=1)
        self.word_embed = nn.Embedding(config.word_vocab_size, config.word_embed_dim,
                padding_idx=1)
        self.user_embed = nn.Embedding(config.user_size, config.user_embed_dim)
        self.dur_embed = nn.Embedding(config.dur_size, config.dur_embed_dim)
        self.slot_embed = nn.Embedding(config.slot_size // config.class_div, 
                config.slot_embed_dim)
        
        # dimensions according to settings
        self.t_rnn_idim = config.word_embed_dim + int(np.sum(config.tc_conv_fn))
        self.st_rnn_idim = config.word_embed_dim + int(np.sum(config.tc_conv_fn))
        self.sm_conv1_idim = (config.user_embed_dim + config.slot_embed_dim +
                config.st_rnn_hdim * 2)
                #config.st_rnn_hdim * 1)
        self.sm_conv2_idim = int(np.sum(config.sm_conv_fn[:3]))
        self.it_idim = (config.user_embed_dim + config.dur_embed_dim +
                config.t_rnn_hdim * 2)
                #config.t_rnn_hdim * 1)
        self.mt_idim = (self.it_idim + int(np.sum(config.sm_conv_fn[3:]))
                 + config.sm_day_num*config.sm_slot_num)

        # convolution layers
        self.tc_conv = nn.ModuleList([nn.Conv2d(
                config.char_embed_dim, config.tc_conv_fn[i],
                (config.tc_conv_fh[i], config.tc_conv_fw[i]),
                stride=1) for i in range(len(config.tc_conv_fn))]) # TODO: padding
        self.sm_conv1 = nn.ModuleList([nn.Conv2d(
                self.sm_conv1_idim, config.sm_conv_fn[i],
                (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                stride=1, padding=config.sm_conv_pd[i]) 
                for i in range(0, len(config.sm_conv_fn)//2)])
        self.sm_mp1 = nn.MaxPool2d(2)
        self.sm_conv2 = nn.ModuleList([nn.Conv2d(
                self.sm_conv2_idim, config.sm_conv_fn[i+3], # added 3 manually
                (config.sm_conv_fh[i], config.sm_conv_fw[i]),
                stride=1, padding=config.sm_conv_pd[i])
                for i in range(len(config.sm_conv_fn)//2)])
        self.sm_mp2 = nn.MaxPool2d(2)

        # rnn layers
        self.t_rnn = nn.LSTM(self.t_rnn_idim, config.t_rnn_hdim, config.t_rnn_ln,
                dropout=config.t_rnn_dr, batch_first=True, bidirectional=True)
        self.st_rnn = nn.LSTM(self.st_rnn_idim, config.st_rnn_hdim, config.st_rnn_ln,
                dropout=config.st_rnn_dr, batch_first=True, bidirectional=True)

        # linear layers
        self.it_nonl = nn.Linear(self.it_idim, self.it_idim)
        self.it_gate = nn.Linear(self.it_idim, self.it_idim)
        self.mt_nonl = nn.Linear(self.mt_idim, self.mt_idim)
        self.mt_gate = nn.Linear(self.mt_idim, self.mt_idim)
        # self.output_fc1 = nn.Linear(self.mt_idim, config.fc1_dim)
        self.output_fc1 = nn.Linear(config.fc1_dim * 2,
                config.sm_day_num * config.sm_slot_num)

        self.day_fc1 = nn.Linear(self.mt_idim, config.fc1_dim)
        self.day_fc2 = nn.Linear(config.fc1_dim, config.sm_day_num)
        self.slot_fc1 = nn.Linear(self.mt_idim, config.fc1_dim)
        self.slot_fc2 = nn.Linear(config.fc1_dim, config.sm_slot_num)

        # initialization
        self.init_word_embed(widx2vec)
        # self.init_params()
        params = self.model_params(debug=False)
        self.optimizer = optim.Adam(params, lr=config.lr) #, weight_decay=config.wd)
        self.criterion = nn.CrossEntropyLoss()
        if config.summary and iter is not None:
            summary_path = 'runs/' + config.model_name + '_' + str(iter)
            self.train_writer = SummaryWriter(summary_path + '/train')
            self.valid_writer = SummaryWriter(summary_path + '/valid')
            self.test_writer = SummaryWriter(summary_path + '/test')

    def init_word_embed(self, widx2vec):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(widx2vec)))
        self.word_embed.weight.requires_grad = False

    def init_params(self):
        for conv in self.tc_conv:
            conv.bias.data = torch.zeros(conv.bias.data.size()).cuda()
        for conv in self.sm_conv1:
            conv.bias.data = torch.zeros(conv.bias.data.size()).cuda()
        for conv in self.sm_conv2:
            conv.bias.data = torch.zeros(conv.bias.data.size()).cuda()

    def model_params(self, debug=True):
        print('model parameters: ', end='')
        params = []
        total_size = 0
        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s\n' % '{:,}'.format(total_size))
        return params
    
    def init_t_rnn_h(self, batch_size):
        return (Variable(torch.zeros(
            self.config.t_rnn_ln*2, batch_size, self.config.t_rnn_hdim)).cuda(),
                Variable(torch.zeros(
            self.config.t_rnn_ln*2, batch_size, self.config.t_rnn_hdim)).cuda())
        """
        return (Variable(torch.zeros(
            self.config.t_rnn_ln*1, batch_size, self.config.t_rnn_hdim)).cuda(),
                Variable(torch.zeros(
            self.config.t_rnn_ln*1, batch_size, self.config.t_rnn_hdim)).cuda())
        """

    def init_st_rnn_h(self, batch_size):
        return (Variable(torch.zeros(
            self.config.st_rnn_ln*2, batch_size, self.config.st_rnn_hdim)).cuda(),
                Variable(torch.zeros(
            self.config.st_rnn_ln*2, batch_size, self.config.st_rnn_hdim)).cuda())
        """
        return (Variable(torch.zeros(
            self.config.st_rnn_ln*1, batch_size, self.config.st_rnn_hdim)).cuda(),
                Variable(torch.zeros(
            self.config.st_rnn_ln*1, batch_size, self.config.st_rnn_hdim)).cuda())
        """

    def as_tensor(self, inputs):
        # assume batch size of 1 when using nets
        torch_inputs = []
        for k, i in enumerate(inputs):
            if k == 5: # for stitle
                for ii in i[0]:
                    ii = Variable(torch.LongTensor(ii).cuda())
                    torch_inputs.append(ii)
            else:
                try: 
                    i = torch.squeeze(Variable(torch.LongTensor(i).cuda()))
                except RuntimeError:
                    i = torch.squeeze(Variable(torch.LongTensor([]).cuda()))
                torch_inputs.append(i)

        return torch_inputs

    def title_layer(self, tc, tw, tl, mode='t'):
        if len(tc.size()) == 0 and mode == 'st':
            return Variable(torch.zeros(1, self.config.st_rnn_hdim * 2).cuda())
            #return Variable(torch.zeros(1, self.config.st_rnn_hdim * 1).cuda())

        # character embedding for title character
        tc_embed = self.char_embed(tc.view(-1, self.config.max_wordlen))
        tc_embed = torch.transpose(torch.unsqueeze(tc_embed, 2), 1, 3).contiguous()
        conv_result = []
        for i, conv in enumerate(self.tc_conv):
            tc_conv = torch.squeeze(conv(tc_embed))
            tc_mp = torch.max(torch.tanh(tc_conv), 2)[0]
            tc_mp = tc_mp.view(-1, self.config.max_sentlen, tc_mp.size(1))
            conv_result.append(tc_mp)
        conv_result = torch.cat(conv_result, 2)

        # word embedding for title, then concat
        tw_embed = self.word_embed(tw.view(-1, self.config.max_sentlen))
        lstm_input = torch.cat((conv_result, tw_embed), 2)

        """
        tmp_zero = Variable(torch.zeros(conv_result.size()).cuda())
        lstm_input = torch.cat((tmp_zero, tw_embed), 2)
        """

        # for input title
        if mode == 't':
            init_t_rnn_h = self.init_t_rnn_h(lstm_input.size(0))
            lstm_out, _ = self.t_rnn(lstm_input, init_t_rnn_h)
            #lstm_out = lstm_out.contiguous().view(-1, self.config.t_rnn_hdim * 1)
            lstm_out = lstm_out.contiguous().view(-1, self.config.t_rnn_hdim * 2)
            lstm_out = lstm_out.cpu()
            
            # select by length
            fw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen + tl.data.cpu() - 1)
            bw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen)
            selected_fw = lstm_out[fw_tl,:].view(
                    -1, self.config.t_rnn_hdim * 2)[:, self.config.t_rnn_hdim:]
            selected_bw = lstm_out[bw_tl,:].view(
                    -1, self.config.t_rnn_hdim * 2)[:, :self.config.t_rnn_hdim]
            selected = torch.cat((selected_fw, selected_bw), 1).cuda()
            #selected = selected_fw.cuda()
            """
            fw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen + tl.data.cpu() - 1)
            selected = lstm_out[fw_tl,:].view(-1, self.config.t_rnn_hdim).cuda()
            """

        # for snapshot title
        elif mode == 'st':
            init_st_rnn_h = self.init_st_rnn_h(lstm_input.size(0))
            lstm_out, _ = self.st_rnn(lstm_input, init_st_rnn_h)
            #lstm_out = lstm_out.contiguous().view(-1, self.config.st_rnn_hdim * 1)
            lstm_out = lstm_out.contiguous().view(-1, self.config.st_rnn_hdim * 2)
            lstm_out = lstm_out.cpu()
            
            # select by length
            fw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen + tl.data.cpu() - 1)
            bw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen)
            selected_fw = lstm_out[fw_tl,:].view(
                    -1, self.config.st_rnn_hdim * 2)[:, self.config.st_rnn_hdim:]
            selected_bw = lstm_out[bw_tl,:].view(
                    -1, self.config.st_rnn_hdim * 2)[:, :self.config.st_rnn_hdim]
            selected = torch.cat((selected_fw, selected_bw), 1).cuda()
            #selected = selected_fw.cuda()
            """
            fw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                    self.config.max_sentlen + tl.data.cpu() - 1)
            selected = lstm_out[fw_tl,:].view(-1, self.config.st_rnn_hdim).cuda()
            """

        return selected 

    def intention_layer(self, user, dur, title, snapshot=None):
        # Highway network on concat
        concat = torch.cat((user, dur, title), 1)
        nonl = F.relu(self.it_nonl(concat))
        gate = F.sigmoid(self.it_gate(concat))
        z = torch.mul(gate, nonl) + torch.mul(1-gate, concat)
        
        """
        tmp_zero = Variable(torch.zeros(1, 60).cuda())
        z = torch.cat((tmp_zero, title), 1)
        """
        return z

    def snapshot_layer(self, user_embed, dur, title, slot):
        # ready for snapshot (contents)
        total_slots = self.config.sm_day_num * self.config.sm_slot_num
        saved_slot = []
        if len(dur.size()) > 0:
            dur = dur / 30 - 1
            new_slot = []
            new_title = []
            for i, (d, s) in enumerate(
                    zip(dur.data.cpu().numpy(), slot.data.cpu().numpy())):
                new_slot.append(s)
                new_title.append(title[i])
                for k in range(d):
                    if s+k+1 < total_slots:
                        new_slot.append(s+k+1)
                        new_title.append(title[i])
            new_slot = np.array(list(new_slot))
            saved_slot = new_slot[:]
            new_slot = Variable(torch.LongTensor(new_slot).cuda())
            new_title = torch.cat(new_title, 0).view(-1, self.config.st_rnn_hdim * 2)
            #new_title = torch.cat(new_title, 0).view(-1, self.config.st_rnn_hdim * 1)
            slot_embed = self.slot_embed(new_slot).view(-1,self.config.slot_embed_dim)
            # slot_embed = Variable(torch.zeros(slot_embed.size()).cuda())
            user_src_embed = user_embed.expand(slot_embed.size(0), user_embed.size(1))
            snapshot_contents = torch.cat((new_title, user_src_embed, slot_embed),1)
            # print('source', snapshot_contents.size())
        saved_slot = Variable(torch.LongTensor(saved_slot).cuda())

        # ready for slot, user embed (base)
        slot_all = Variable(torch.arange(0, total_slots).type(torch.LongTensor).cuda())
        slot_all_embed = self.slot_embed(slot_all)
        user_all_embed = user_embed.expand(slot_all_embed.size(0), user_embed.size(1))
        zero_concat = torch.zeros(total_slots, self.config.st_rnn_hdim * 2)
        #zero_concat = torch.zeros(total_slots, self.config.st_rnn_hdim * 1)
        snapshot_base = torch.cat((zero_concat, 
            user_all_embed.data.cpu(), slot_all_embed.data.cpu()), 1)
        # print('base', snapshot_base.size()) 

        # ready for snapshot map (empty)
        snapshot_map = Variable(torch.zeros(total_slots, self.sm_conv1_idim))
        # print('dest', snapshot_map.size())
        
        if len(dur.size()) > 0:
            index = new_slot.data.unsqueeze(1)
            index = index.expand_as(snapshot_contents)
        slot_all = slot_all.data.unsqueeze(1)
        slot_all = slot_all.expand_as(snapshot_base)

        # scatter base and then the contents
        snapshot_map.data.cpu().scatter_(0, torch.LongTensor(slot_all.cpu()), 
                snapshot_base)
        if len(dur.size()) > 0:
            snapshot_map.data.cpu().scatter_(0, torch.LongTensor(index.cpu()), 
                    snapshot_contents.data.cpu())

        # (7, 48, 110)
        snapshot_map = snapshot_map.view(self.config.sm_day_num,
                self.config.sm_slot_num, self.sm_conv1_idim) 
        # (110, 7, 48)
        snapshot_map = torch.transpose(torch.transpose(snapshot_map, 0, 2), 1, 2)

        # multiple filter conv
        conv_list = [self.sm_conv1, self.sm_conv2]
        mp_list = [self.sm_mp1, self.sm_mp2]
        snapshot_mf = torch.unsqueeze(snapshot_map, 0)
        for layer_idx, sm_conv in enumerate(conv_list):
            conv_result = []
            for filter_idx, conv in enumerate(sm_conv):
                conv_out = conv(snapshot_mf.cuda())
                conv_result.append(conv_out)
            snapshot_mf = torch.cat(conv_result, 1)
            if layer_idx < len(conv_list) - 1:
                snapshot_mf = F.relu(snapshot_mf)
            # print('relu', snapshot_mf.size())
            else:
                snapshot_mf = torch.max(snapshot_mf.view(
                    1, snapshot_mf.size(1), -1), 2)[0]
            # snapshot_mf = mp_list[layer_idx](snapshot_mf)
            # print('mp', snapshot_mf.size())

        # snapshot_mf = torch.max(snapshot_mf.view(
        #     1, snapshot_mf.size(1), -1), 2)[0]
        # print(snapshot_mf.size())

        return snapshot_mf, saved_slot

    def matching_layer(self, intention, snapshot_mf, grid):
        # Highway network for mf
        grid_vec = Variable(torch.FloatTensor(
                self.config.sm_day_num*self.config.sm_slot_num).zero_())
        if not len(grid.size()) == 0:
            grid_vec.data.cpu().scatter_(0, torch.LongTensor(grid.data.cpu()), 1)
        grid_vec = torch.unsqueeze(grid_vec, 0).cuda()
        concat = torch.cat((intention, snapshot_mf, grid_vec), 1)
        # concat = torch.cat((intention, snapshot_mf), 1)
        nonl = F.relu(self.mt_nonl(concat))
        gate = F.sigmoid(self.mt_gate(concat))
        z = torch.mul(gate, nonl) + torch.mul(1-gate, concat)

        # (it_idim * 2 + mt_idim * 1)
        output = z
        # output_fc1 = F.relu(self.output_fc1(output))
        # output = self.output_fc2(output_fc1)

        day_fc1 = F.relu(self.day_fc1(output))
        slot_fc1 = F.relu(self.slot_fc1(output))
        output = self.output_fc1(torch.cat((day_fc1, slot_fc1), 1))
        return output

    def forward(self, inputs):
        """
        inputs (batch_size = 1)
            - user: [batch]
            - dur: [batch]
            - tc: [batch, sentlen, wordlen]
            - tw: [batch, sentlen]
            - tl: [batch]
            - stitle: [batch, snum], ((sentlen, wordlen), (sentlen), (1))
            - sdur: [batch, snum]
            - sslot: [batch, snum]

        """
        user, dur, tc, tw, tl, stc, stw, stl, sdur, sslot, gr = self.as_tensor(inputs)
        user_embed = self.user_embed(user).view(-1, self.config.user_embed_dim)
        # user_embed = Variable(torch.zeros(user_embed.size()).cuda())
        dur_embed = self.dur_embed(dur).view(-1, self.config.dur_embed_dim)
        # dur_embed = Variable(torch.zeros(dur_embed.size()).cuda())
        # print('user duration embed', user_embed.size(), dur_embed.size())

        title_rep = self.title_layer(tc, tw, tl)
        # print('title layer', title_rep.size())

        intention_rep = self.intention_layer(user_embed, dur_embed, title_rep)
        # print('intention layer', intention_rep.size())

        if not self.config.no_snapshot:
            stitle_rep = self.title_layer(stc, stw, stl, mode='st')
            # print('stitle layer', stitle_rep.size())

            mf, grid = self.snapshot_layer(user_embed, sdur, stitle_rep, sslot)
            # print('snapshot layer', snapshot_rep.size())

            output = self.matching_layer(intention_rep, mf, gr) # or gr
            # print('matching layer', matching.size())
        else:
            output_fc1 = F.relu(self.output_fc1(intention_rep))
            output = self.output_fc2(output_fc1)

        assert output.size(1) == self.config.sm_day_num * self.config.sm_slot_num
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

    def get_metrics(self, outputs, targets, ex_targets=None, topk=5):
        max_idx = torch.max(outputs, 1)[1].data.cpu().numpy()
        # outputs_topk = torch.topk(outputs, topk)[1].data.cpu().numpy()
        targets_value = torch.cat([o[t] for o, t in zip(outputs, targets)])
        targets = targets.data.cpu().numpy()
        targets_value = targets_value.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()

        def exclusive_topk(output, target, ex_target, topk):
            # assert np.amin(output, 0) > -99999
            output[ex_target] = -float("inf")
            max_idx = output.argsort()[-topk:][::-1]
            for mi in max_idx:
                assert mi not in ex_target
            return max_idx
        outputs_topk = [exclusive_topk(o[:], t, et, topk) for (o, t, et)
                in zip(outputs, targets, ex_targets)]
        
        ex1 = np.mean([float(k == tk[0]) for (k, tk)
            in zip(targets, outputs_topk)]) * 100
        ex5 = np.mean([float(k in tk) for (k, tk)
            in zip(targets, outputs_topk)]) * 100
        fb1 = np.mean([float(np.absolute(k - m) <= 1) for (k, m)
            in zip(targets, max_idx)]) * 100
        fb2 = np.mean([float(np.absolute(k - m) <= 2) for (k, m)
            in zip(targets, max_idx)]) * 100
        mrr = np.mean([1/np.sum(t <= out) for t, out 
            in zip(targets_value, outputs)]) * 100

        def square_distance1(target, pred):
            target_x = target // 24
            target_y = target % 24
            pred_x = pred // 24
            pred_y = pred % 24
            return float(np.absolute(target_x-pred_x) <= 1 and
                    np.absolute(target_y-pred_y) <= 1)

        sq1 = np.sum([square_distance1(k, m) for (k, m)
            in zip(targets, max_idx)])

        def euclidean_distance(target, pred):
            target_x = target // 24
            target_y = target % 24
            pred_x = pred // 24
            pred_y = pred % 24
            return math.sqrt(np.absolute(target_x-pred_x)**2 +
                    np.absolute(target_y-pred_y)**2)

        euc = np.mean([1/(euclidean_distance(k, m)+1) for (k, m)
            in zip(targets, max_idx)]) * 100

        return ex1, ex5, mrr, euc
 
    def save_checkpoint(self, state, filename=None):
        if filename is None:
            filename = self.config.checkpoint_dir + self.config.model_name + '.pth'
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = self.config.checkpoint_dir + self.config.model_name + '.pth'
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> load checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.config = checkpoint['config']

    def write_summary(self, mode, metrics, offset):
        if mode == 'tr':
            writer = self.train_writer
        elif mode == 'va':
            writer = self.valid_writer
        elif mode == 'te':
            writer = self.test_writer
        writer.add_scalar('metric/loss', metrics[0], offset)
        writer.add_scalar('metric/acc1', metrics[1], offset)
        writer.add_scalar('metric/acc5', metrics[2], offset)
        writer.add_scalar('metric/mrr', metrics[3], offset)
        writer.add_scalar('metric/euc', metrics[4], offset)

    def close_summary_writer(self):
        self.train_writer.close()
        self.valid_writer.close()
        self.test_writer.close()

