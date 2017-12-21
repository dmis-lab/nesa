import argparse
import dataset
import math
from model import NETS
import numpy as np
import os
import pickle
# import pprint
import torch


def get_dataset(_config, pretrained_dict_path='./data/nets_sm_w0_dict.pkl'):
    if os.path.exists(_config.preprocess_load_path):
        print('Found existing test dataset', _config.preprocess_load_path)
        _test_dataset = pickle.load(open(_config.preprocess_load_path, 'rb'))
    else:
        print('Creating new test dataset..', )
        nets_dictionary = pickle.load(open(pretrained_dict_path, 'rb'))
        _test_dataset = dataset.Dataset(_config,
                                        pretrained_dict=nets_dictionary,
                                        test_mode=True)
        if len(_test_dataset.test_data) == 0:
            print('no events')
            return None
        pickle.dump(_test_dataset, open(_config.preprocess_save_path, 'wb'))
    return _test_dataset


def get_model(_dataset, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    ckpt_config = checkpoint['config']
    ckpt_config.load_path = ckpt_path
    ckpt_config.word_dr = 0.5
    _dataset.config.__dict__.update(ckpt_config.__dict__)

    _model = NETS(ckpt_config, _dataset.widx2vec).cuda()
    _model.config.checkpoint_dir = './data/'
    _model.load_checkpoint()
    # pprint.PrettyPrinter().pprint(_model.config.__dict__)
    return _model


def measure_performance(_dataset, _nets_model, shuffle=False, topk=5):
    def exclusive_topk(output, ex_target, _topk):
        assert np.amin(output, 0) > -99999
        output[ex_target] -= 99999
        _max_idx = output.argsort()[-_topk:][::-1]
        for mi in _max_idx:
            assert mi not in ex_target
        return _max_idx

    def euclidean_distance(target, pred):
        target_x = target // 24
        target_y = target % 24
        pred_x = pred // 24
        pred_y = pred % 24
        return math.sqrt(np.absolute(target_x - pred_x) ** 2 +
                         np.absolute(target_y - pred_y) ** 2)
    if shuffle:
        _dataset.shuffle_data(mode='te')

    performance_dict = dict()
    performance_dict['acc1'] = 0.
    performance_dict['acc5'] = 0.
    performance_dict['mrr'] = 0.
    performance_dict['ieuc'] = 0.
    performance_dict['count'] = 0

    while True:
        inputs, targets = _dataset.get_next_batch(batch_size=1, mode='te',
                                                  pad=True)

        outputs, reps = _nets_model(_dataset.parse_inputs(inputs))

        ex_targets = [i[4] for i in inputs]
        max_idx = torch.topk(outputs, 1)[1].data.cpu().numpy()
        targets_value = torch.cat([o[t] for o, t in zip(outputs, targets)])
        targets_value = targets_value.data.cpu().numpy()
        _outputs = outputs.data.cpu().numpy()

        outputs_topk = [exclusive_topk(o[:], et, topk) for (o, t, et)
                        in zip(_outputs, targets, ex_targets)]

        performance_dict['acc1'] += np.sum([float(k == tk[0]) for (k, tk)
                                            in zip(targets, outputs_topk)])
        performance_dict['acc5'] += np.sum([float(k in tk) for (k, tk)
                                            in zip(targets, outputs_topk)])
        performance_dict['mrr'] += np.sum([1 / np.sum(t <= out)
                                           for t, out
                                           in zip(targets_value, _outputs)])
        performance_dict['ieuc'] += \
            np.sum([1 / (euclidean_distance(k, m) + 1)
                    for (k, m) in zip(targets, max_idx)])
        performance_dict['count'] += 1

        if (_dataset.test_ptr + 1) % 1000 == 0:
            print(_dataset.test_ptr)

        if _dataset.test_ptr == 0:
            # print('\ntest iteration end')
            break

    acc1 = performance_dict['acc1'] / performance_dict['count']
    acc5 = performance_dict['acc5'] / performance_dict['count']
    mrr = performance_dict['mrr'] / performance_dict['count']
    ieuc = performance_dict['ieuc'] / performance_dict['count']

    print('acc1 %.4f' % acc1)
    print('acc5 %.4f' % acc5)
    print('mrr  %.4f' % mrr)
    print('ieuc %.4f' % ieuc)
    print('#events', performance_dict['count'])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_path", type=str,
                            default='./data/sample_events.csv')
    arg_parser.add_argument("--preprocess_path", type=str,
                            default='./data/preprocess(test).pkl')
    args = arg_parser.parse_args()

    config = dataset.Config()
    config.test_path = args.test_path
    config.preprocess_save_path = args.preprocess_path
    config.preprocess_load_path = args.preprocess_path

    print('Loading test dataset')
    test_dataset = get_dataset(config)
    assert test_dataset is not None

    print('Loading NETS model')
    nets_model = get_model(test_dataset, './data/conv_nets.pth')

    print('Measuring NETS performance on test data')
    measure_performance(test_dataset, nets_model)
