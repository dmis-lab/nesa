import numpy as np
# import sys
import csv
import pprint
import copy
import pickle
import nltk
import string
from os.path import expanduser

nltk.download('punkt')


class Dataset(object):
    def __init__(self, config, pretrained_dict=None, test_mode=False):
        self.config = config
        self.initial_settings()
        self.initialize_dictionary()

        # self.build_word_dict(self.config.train_path)
        # self.build_word_dict(self.config.valid_path)
        # self.build_word_dict(self.config.test_path, update=False)
        # self.get_pretrained_word(self.config.word2vec_path)

        if pretrained_dict is not None and test_mode:
            # word
            self.word2idx = pretrained_dict['word2idx']
            self.idx2word = pretrained_dict['idx2word']
            self.widx2vec = pretrained_dict['widx2vec']

            # char
            self.char2idx = pretrained_dict['char2idx']
            self.idx2char = pretrained_dict['idx2char']

            # duration
            self.dur2idx = pretrained_dict['dur2idx']
            self.idx2dur = pretrained_dict['idx2dur']

            # user
            self.user2idx = pretrained_dict['user2idx']
            self.idx2user = pretrained_dict['idx2user']

            # max len
            self.config.max_sentlen = pretrained_dict['config.max_sentlen']
            self.config.max_wordlen = pretrained_dict['config.max_wordlen']

            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)
            self.config.user_size = len(self.user2idx)
            self.config.dur_size = len(self.dur2idx)
            self.config.slot_size = self.slot_size
            self.config.class_div = self.class_div

        # self.train_data = self.process_data(
        #         self.config.train_path,
        #         update_dict=True)
        # self.valid_data = self.process_data(
        #         self.config.valid_path,
        #         update_dict=True)
        self.test_data = self.process_data(
                self.config.test_path, test_mode=test_mode)

        self.train_ptr = 0
        self.valid_ptr = 0
        self.test_ptr = 0
    
    def initial_settings(self):
        # predefined settings
        self.UNK = 'UNK'
        self.PAD = 'PAD'
        self.feature_len = 11
        self.duration_unit = 30 # min
        self.max_rs_dist = 2 # reg-st week distance
        self.class_div = 2 # 168 output
        self.slot_size = 336
        self.max_snapshot = float("inf") # 35
        self.min_word_cnt = 0
        self.max_title_len = 50
        self.max_event_cnt = 5000

    def initialize_dictionary(self):
        # dictionary specific settings
        self.char2idx = {}
        self.idx2char = {}
        self.word2idx = {}
        self.idx2word = {}
        self.widx2vec = []  # pretrained
        self.user2idx = {}
        self.idx2user = {}
        self.dur2idx = {}
        self.idx2dur = {}
        self.char2idx[self.UNK] = 0
        self.char2idx[self.PAD] = 1
        self.idx2char[0] = self.UNK
        self.idx2char[1] = self.PAD
        self.word2idx[self.UNK] = 0
        self.word2idx[self.PAD] = 1
        self.idx2word[0] = self.UNK
        self.idx2word[1] = self.PAD
        self.user2idx[self.UNK] = 0
        self.idx2user[0] = self.UNK
        self.dur2idx[self.UNK] = 0
        self.idx2dur[0] = self.UNK
        self.initial_word_dict = {}
        self.invalid_weeks = []
        self.user_event_cnt = {}
    
    def update_dictionary(self, key, mode=None):
        # update dictionary given a key
        if mode == 'c':
            if key not in self.char2idx:
                self.char2idx[key] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = key
        elif mode == 'w':
            if key not in self.word2idx:
                self.word2idx[key] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = key
        elif mode == 'u':
            if key not in self.user2idx:
                self.user2idx[key] = len(self.user2idx)
                self.idx2user[len(self.idx2user)] = key
        elif mode == 'd':
            if key not in self.dur2idx:
                self.dur2idx[key] = len(self.dur2idx)
                self.idx2dur[len(self.idx2dur)] = key
    
    def map_dictionary(self, key_list, dictionary, reverse=False):
        # mapping list of keys into dictionary 
        #   reverse=False : word2idx, char2idx
        #   reverse=True : idx2word, idx2char
        output = []
        for key in key_list:
            if key in dictionary:
                if reverse and key == 1: # skip PAD for reverse
                    continue
                else:
                    output.append(dictionary[key])
            else: # unknown key
                if not reverse:
                    output.append(dictionary[self.UNK])
                else:
                    output.append(dictionary[0])
        return output
    
    def build_word_dict(self, path, update=True):
        print('### build word dict %s' % path)
        with open(path, 'r', newline='', encoding='utf-8') as f:
            calendar_data = csv.reader(f, quotechar='"')
            prev_what_list = []
            prev_week_key = ''
            for k, features in enumerate(calendar_data):
                assert len(features) == self.feature_len 
                what = features[1]
                user_id = features[0]
                st_year = features[5]
                st_week = features[6]
                reg_seq = int(features[7])
                week_key = '_'.join([user_id, st_year, st_week])

                def check_printable(text, w_key):
                    for char in text:
                        if char not in string.printable:
                            if w_key not in self.invalid_weeks:
                                self.invalid_weeks.append(w_key)
                            return False
                    return True
                
                def check_maxlen(text, w_key):
                    what_split = nltk.word_tokenize(text)
                    if len(what_split) > self.max_title_len:
                        if w_key not in self.invalid_weeks:
                            self.invalid_weeks.append(w_key)
                            return False
                    return True

                if reg_seq == 0:
                    assert prev_week_key != week_key
                    # process previous week's what list
                    if prev_week_key not in self.invalid_weeks and update:
                        for single_what in prev_what_list:
                            what_split = nltk.word_tokenize(single_what)
                            if self.config.word2vec_type == 6:
                                what_split = [w.lower() for w in what_split]
                            for word in what_split:
                                if word not in self.initial_word_dict:
                                    self.initial_word_dict[word] = (
                                            len(self.initial_word_dict), 1)
                                else:
                                    self.initial_word_dict[word] = (
                                            self.initial_word_dict[word][0],
                                            self.initial_word_dict[word][1] + 1)

                    # first event should be also printable
                    if check_printable(what, week_key) \
                            and check_maxlen(what, week_key):
                        prev_what_list = [what]
                    else:
                        prev_what_list = []
                    prev_week_key = week_key
                else:
                    assert prev_week_key == week_key, \
                        '%s != %s' % (prev_week_key, week_key)
                    if prev_week_key in self.invalid_weeks:
                        continue
                    
                    # event title should be printable
                    if check_printable(what, prev_week_key) \
                            and check_maxlen(what, prev_week_key):
                        prev_what_list.append(what)

        print('initial dict size', len(self.initial_word_dict))

    def get_pretrained_word(self, path):
        print('\n### load pretrained %s' % path)
        word2vec = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split(' ')
                word2vec[cols[0]] = [float(l) for l in cols[1:]]
        
        widx2vec = []
        unk_cnt = 0
        widx2vec.append([0.0] * self.config.word_embed_dim) # UNK
        widx2vec.append([0.0] * self.config.word_embed_dim) # PAD

        for word, (word_idx, word_cnt) in self.initial_word_dict.items():
            if word != 'UNK' and word !='PAD':
                assert word_cnt > 0
                if word in word2vec and word_cnt > self.min_word_cnt:
                    self.update_dictionary(word, 'w')
                    widx2vec.append(word2vec[word])
                else:
                    unk_cnt += 1
        # print('apple:', self.word2idx['apple'], word2vec['apple'][:5])
        # print('apple:', widx2vec[self.word2idx['apple']][:5])
        print('pretrained vectors', np.asarray(widx2vec).shape, 'unk', unk_cnt)
        print('dictionary change', len(self.initial_word_dict), 
                'to', len(self.word2idx), len(self.idx2word), end='\n\n')

        self.widx2vec = widx2vec

    def process_data(self, path, update_dict=False, test_mode=False):
        print('### processing %s' % path)
        total_data = []
        max_wordlen = max_sentlen = max_dur = max_snapshot = 0
        min_dur = float("inf")

        with open(path, 'r', newline='', encoding='utf-8') as f:
            """
            Each line consists of features below:
                0: user id
                1: what
                2: duration (minute)
                3: register time
                4: start time
                5: start year
                6: start week
                7: register sequence in the week
                8: register start week distance
                9: register start day distance
                10: start time slot (y)
            """
            prev_user = ''
            prev_st_yw = ('', '')
            saved_snapshot = []
            calendar_data = csv.reader(f, quotechar='"')

            for k, features in enumerate(calendar_data):
                assert len(features) == self.feature_len
                user_id = features[0]
                what = features[1]
                duration = int(features[2])
                reg_time = features[3]
                st_time = features[4]
                st_year = features[5]
                st_week = features[6]
                reg_seq = int(features[7])
                reg_st_week_dist = int(features[8])
                reg_st_day_dist = int(features[9])
                st_slot = int(features[10])

                # remove unprintable weeks
                week_key = '_'.join([user_id, st_year, st_week])
                if week_key in self.invalid_weeks:
                    continue

                # ready for one week data
                curr_user = user_id
                curr_st_yw = (st_year, st_week)

                # filter user by event count
                if user_id in self.user_event_cnt:
                    if self.user_event_cnt[user_id] > self.max_event_cnt:
                        prev_user = curr_user
                        prev_st_yw = curr_st_yw
                        continue

                # ignore data that was written in future
                if reg_st_week_dist < 0:
                    prev_user = curr_user
                    prev_st_yw = curr_st_yw
                    continue

                if test_mode:
                    input_user = self.user2idx[self.UNK]
                else:
                    # process user feature
                    if update_dict:
                        self.update_dictionary(user_id, 'u')
                    if user_id not in self.user2idx:
                        prev_user = curr_user
                        prev_st_yw = curr_st_yw
                        continue
                    input_user = self.user2idx[user_id]
                # print('input_user', input_user)

                # process title feature
                what_split = nltk.word_tokenize(what)
                if self.config.word2vec_type == 6:
                    what_split = [w.lower() for w in what_split]
                for word in what_split:
                    max_wordlen = (len(word)
                            if len(word) > max_wordlen else max_wordlen)
                max_sentlen = (len(what_split)
                        if len(what_split) > max_sentlen else max_sentlen)

                if update_dict:
                    for char in what:
                        self.update_dictionary(char, 'c')
                if max_wordlen > self.config.max_wordlen:
                    self.config.max_wordlen = max_wordlen
                if max_sentlen > self.config.max_sentlen:
                    self.config.max_sentlen = max_sentlen
                
                sentchar = []
                for word in what_split:
                    sentchar.append(self.map_dictionary(word, self.char2idx))
                sentword = self.map_dictionary(what_split, self.word2idx)
                length = len(sentword)
                assert len(sentword) == len(sentchar)
                input_title = [sentchar, sentword, length]

                # process duration feature
                max_dur = max_dur if max_dur > duration else duration
                min_dur = min_dur if min_dur < duration else duration
                fine_duration = (duration//self.duration_unit) * self.duration_unit
                fine_duration += (int(duration % self.duration_unit > 0) *
                        self.duration_unit)
                if duration % self.duration_unit == 0:
                    assert duration == fine_duration
                else:
                    assert fine_duration - duration < self.duration_unit

                if update_dict:
                    self.update_dictionary(fine_duration, 'd')
                input_duration = self.dur2idx[fine_duration]

                # TODO: process reg_time feature

                # process st_slot feature
                assert st_slot < self.slot_size
                input_slot = st_slot // self.class_div
                target_slot = st_slot // self.class_div
                
                # process snapshot
                if reg_seq == 0: # start of a new week
                    assert (curr_user != prev_user) or (curr_st_yw != prev_st_yw)
                    prev_user = curr_user
                    prev_st_yw = curr_st_yw
                    prev_grid = []
                    input_snapshot = []
                    saved_snapshot = [[input_title, fine_duration, input_slot]]
                else: # same as the prev week
                    assert (curr_user == prev_user) and (curr_st_yw == prev_st_yw)
                    # input_snapshot = copy.deepcopy(saved_snapshot)
                    prev_grid = [svs[2] for svs in saved_snapshot] 
                    if input_slot in prev_grid:
                        continue
                    input_snapshot = saved_snapshot[:]
                    saved_snapshot.append([input_title, fine_duration, input_slot]) 

                # transform snapshot features into slot grid
                input_grid = [ips[2] for ips in input_snapshot]

                # filter by register distance & max_snapshot
                if (reg_st_week_dist <= self.max_rs_dist
                        and len(input_snapshot) <= self.max_snapshot):
                    max_snapshot = (max_snapshot if max_snapshot>len(input_snapshot)
                            else len(input_snapshot))
                    total_data.append([input_user, input_title, input_duration,
                        input_snapshot, input_grid, target_slot])

                    if user_id not in self.user_event_cnt:
                        self.user_event_cnt[user_id] = 1
                    else:
                        self.user_event_cnt[user_id] += 1

        if update_dict:
            self.config.char_vocab_size = len(self.char2idx)
            self.config.word_vocab_size = len(self.word2idx)
            self.config.user_size = len(self.user2idx)
            self.config.dur_size = len(self.dur2idx)
            self.config.slot_size = self.slot_size
            self.config.class_div = self.class_div

        print('data size', len(total_data))
        print('max duration', max_dur)
        print('min duration', min_dur)
        print('max snapshot', max_snapshot)
        print('max wordlen', max_wordlen)
        print('max sentlen', max_sentlen, end='\n\n')

        return total_data

    def pad_sent_word(self, sentchar, sentword):
        # pad sentword
        assert len(sentword) <= self.config.max_sentlen, '%d > %d' % (len(sentword), self.config.max_sentlen)
        while len(sentword) != self.config.max_sentlen:
            sentword.append(self.word2idx[self.PAD])
        # pad word in sentchar
        for word in sentchar:
            assert len(word) <= self.config.max_wordlen, '%d > %d' % (len(word), self.config.max_wordlen)
            while len(word) != self.config.max_wordlen:
                word.append(self.char2idx[self.PAD])
        # pad sentchar
        assert len(sentchar) <= self.config.max_sentlen, '%d > %d' % (len(sentchar), self.config.max_sentlen)
        while len(sentchar) != self.config.max_sentlen:
            sentchar.append([self.char2idx[self.PAD]] * self.config.max_wordlen)
        assert len(sentchar) == len(sentword)

    def pad_data(self, dataset):
        for data in dataset:
            _, title, _, snapshot, _, _ = data
            sentchar, sentword, _ = title
            self.pad_sent_word(sentchar, sentword)
            for title, _, _ in snapshot:
                sentchar, sentword, _ = title
                self.pad_sent_word(sentchar, sentword)

        return dataset
    
    def get_next_batch(self, mode='tr', batch_size=None, pad=True):
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if mode == 'tr':
            ptr = self.train_ptr
            data = self.train_data
        elif mode == 'va':
            ptr = self.valid_ptr
            data = self.valid_data
        elif mode == 'te':
            ptr = self.test_ptr
            data = self.test_data
        else:
            raise ValueError('Unknown mode %s' % mode)

        batch_size = (batch_size if ptr+batch_size<=len(data)
                else len(data)-ptr)
        # print(data[ptr:ptr + batch_size][0])
        if pad:
            padded_data = self.pad_data(copy.deepcopy(data[ptr:ptr+batch_size]))
        else:
            padded_data = data[ptr:ptr+batch_size]
        inputs = [d[0:5] for d in padded_data]
        targets = [d[5] for d in padded_data]
        
        if mode == 'tr':
            self.train_ptr = (ptr + batch_size) % len(data)
        elif mode == 'va':
            self.valid_ptr = (ptr + batch_size) % len(data)
        elif mode == 'te':
            self.test_ptr = (ptr + batch_size) % len(data)

        return inputs, targets
    
    def get_batch_ptr(self, mode):
        if mode == 'tr':
            return self.train_ptr
        elif mode == 'va':
            return self.valid_ptr
        elif mode == 'te':
            return self.test_ptr

    def get_dataset_len(self, mode):
        if mode == 'tr':
            return len(self.train_data)
        elif mode == 'va':
            return len(self.valid_data)
        elif mode == 'te':
            return len(self.test_data)

    def initialize_batch_ptr(self, mode=None):
        if mode is None:
            self.train_ptr = 0
            self.valid_ptr = 0
            self.test_ptr = 0
        elif mode == 'tr':
            self.train_ptr = 0
        elif mode == 'va':
            self.valid_ptr = 0
        elif mode == 'te':
            self.test_ptr = 0

    def decode_data(self, data, only_inputs=False, skip_char=False):
        if only_inputs:
            user, title, duration, snapshot, grid = data
        else:
            user, title, duration, snapshot, grid, slot = data
            print('slot', slot)

        user = self.idx2user[user]
        title_c, title_w, l = title
        title_c = [self.map_dictionary(c, self.idx2char, True) for c in title_c]
        title_w = ' '.join(self.map_dictionary(title_w, self.idx2word, True))
        duration = self.idx2dur[duration]
        print('user', user)
        if skip_char:
            print('title', title_w, l)
        else:
            print('title', title_w, title_c, l)
        print('duration', duration)

        ss_set = []
        for kk, (tt, dd, ss) in enumerate(snapshot):
            _, t_w, _ = tt
            t_w = ' '.join(self.map_dictionary(t_w, self.idx2word, True))
            print('snapshot', kk, t_w, str(dd), ss) 
            ss_set.append(ss)
        print('grid', grid)

        return ss_set

    def sample_data(self, idx=None, decode=False, mode='va'):
        if mode == 'tr':
            data = self.train_data
        elif mode == 'va':
            data = self.valid_data
        elif mode == 'te':
            data = self.test_data

        if idx is None:
            idx = np.random.randint(len(data))
        user, title, duration, snapshot, grid, slot = data[idx]
        print('idx', idx)

        if decode:
            self.decode_data(data[idx])
        else:
            print('user', user)
            print('title', title)
            print('duration', duration)
            print('snapshot', snapshot)
            print('grid', grid)
            print('slot', slot)

        return data[idx]

    def shuffle_data(self, mode='all', seed=None):
        if seed is not None:
            np.random.seed(seed)
        if mode == 'all':
            np.random.shuffle(self.train_data)
            np.random.shuffle(self.valid_data)
            np.random.shuffle(self.test_data)
        elif mode == 'tr':
            np.random.shuffle(self.train_data)
        elif mode == 'va':
            np.random.shuffle(self.valid_data)
        elif mode == 'te':
            np.random.shuffle(self.test_data)

    def small_setting(self, value):
        self.train_data = self.train_data[:len(self.train_data) // value]
        self.valid_data = self.valid_data[:len(self.valid_data) // value]
        self.test_data = self.test_data[:len(self.test_data) // value]
        print('train data', len(self.train_data))
        print('valid data', len(self.valid_data))
        print('test data', len(self.test_data), end='\n\n')
    
    def parse_inputs(self, inputs):
        # user and duration
        inputs_user = np.array([i[0] for i in inputs])
        inputs_dur = np.array([i[2] for i in inputs])
        
        # title (char, word, length)
        inputs_title = [i[1] for i in inputs]
        inputs_tc = np.array([t[0] for t in inputs_title])
        inputs_tw = np.array([t[1] for t in inputs_title])
        inputs_tl = np.array([t[2] for t in inputs_title])
        
        # snapshot (title, duration, slot)
        inputs_ss = [i[3] for i in inputs]
        inputs_stitle = []
        inputs_sdur = []
        inputs_sslot = []
        for kk, ss in enumerate(inputs_ss):
            inputs_stc = []
            inputs_stw = []
            inputs_stl = []
            inputs_sd = []
            inputs_ssl = []
            for e in ss:
                inputs_stc.append(e[0][0])
                inputs_stw.append(e[0][1])
                inputs_stl.append(e[0][2])
                inputs_sd.append(e[1])
                inputs_ssl.append(e[2])
            inputs_stitle.append([inputs_stc, inputs_stw, inputs_stl])
            inputs_sdur.append(inputs_sd)
            inputs_sslot.append(inputs_ssl)
        inputs_stitle = np.array(inputs_stitle)
        inputs_sdur = np.array(inputs_sdur)
        inputs_sslot = np.array(inputs_sslot)

        # grid
        inputs_grid = [i[4] for i in inputs]
        
        return (inputs_user, inputs_dur, inputs_tc, inputs_tw, inputs_tl,
                inputs_stitle, inputs_sdur, inputs_sslot, inputs_grid)


class Config(object):
    def __init__(self):
        self.train_path = './data/train.csv'
        self.valid_path = './data/valid.csv'
        self.test_path = './data/test.csv'
        self.word2vec_path = expanduser('~') + '/common/glove/glove.840B.300d.txt'
        self.word2vec_type = 840  # 6 or 840 (B)
        self.word_embed_dim = 300
        self.batch_size = 32
        self.max_wordlen = 0
        self.max_sentlen = 0
        self.char_vocab_size = 0
        self.word_vocab_size = 0
        self.user_size = 0
        self.dur_size = 0
        self.class_div = 0
        self.slot_size = 0
        self.save_preprocess = True
        self.preprocess_save_path = './data/preprocess(tmp).pkl'
        self.preprocess_load_path = './data/preprocess(s35).pkl'


if __name__ == '__main__':
    config = Config()
    if config.save_preprocess:
        dataset = Dataset(config)
        pickle.dump(dataset, open(config.preprocess_save_path, 'wb'))
    else:
        print('## load preprocess %s' % config.preprocess_load_path)
        dataset = pickle.load(open(config.preprocess_load_path, 'rb'))
   
    # dataset config must be valid
    pp = lambda x: pprint.PrettyPrinter().pprint(x)
    pp(([(k,v) for k, v in vars(dataset.config).items() if '__' not in k]))
    print()
    
    print('ptr test!')
    dataset.shuffle_data()
    input, target = dataset.get_next_batch(batch_size=1)
    dataset.decode_data(input[0], only_inputs=True, skip_char=True)
    print(target[0])
    dataset.train_ptr = 0
    input, target = dataset.get_next_batch(batch_size=1)
    dataset.decode_data(input[0], only_inputs=True, skip_char=True)
    print(target[0])
    print()

    print('shuffle test!')
    dataset.shuffle_data()
    dataset.sample_data(idx=0, decode=True, mode='tr')
    dataset.shuffle_data()
    dataset.sample_data(idx=0, decode=True, mode='tr')
   
    while True:
        i, t = dataset.get_next_batch(batch_size=3000, mode='te', pad=True)
        # print(dataset.valid_ptr, len(i))
        if dataset.test_ptr == 0:
            print('\niteration test pass!')
            break
