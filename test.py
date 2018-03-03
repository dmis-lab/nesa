import argparse
import dataset
# from dataset import NETSDataset, Config
from model import NETS
import numpy as np
import os
import pickle
import random
import torch
from torch.autograd import Variable


def get_dataset(_config, pretrained_dict_path):
    print('Creating the test dataset pickle..', )
    nets_dictionary = pickle.load(open(pretrained_dict_path, 'rb'))
    _test_dataset = dataset.NETSDataset(_config, nets_dictionary)
    if len(_test_dataset.test_data) == 0:
        print('no events')
        return None
    pickle.dump(_test_dataset, open(_config.preprocess_save_path, 'wb'))
    return _test_dataset


def get_model(_dataset, model_path):
    model_dir, model_filename = os.path.split(model_path)
    checkpoint = torch.load(model_path)
    ckpt_config = checkpoint['config']
    ckpt_config.load_path = model_path
    _dataset.config.__dict__.update(ckpt_config.__dict__)

    _model = NETS(ckpt_config, _dataset.widx2vec).cuda()
    _model.config.checkpoint_dir = model_dir + '/'
    _model.load_checkpoint(filename=model_filename)
    # import pprint
    # pprint.PrettyPrinter().pprint(_model.config.__dict__)
    return _model


def measure_performance(_dataset, _nets_model, batch_size=1):
    performance_dict = dict()
    performance_dict['recall1'] = 0.
    performance_dict['recall5'] = 0.
    performance_dict['mrr'] = 0.
    performance_dict['ieuc'] = 0.
    performance_dict['count'] = 0
    performance_dict['steps'] = 0.

    _, _, test_loader = _dataset.get_dataloader(batch_size=batch_size)
    for d_idx, ex in enumerate(test_loader):
        outputs, reps = _nets_model(*ex[:-1])
        metrics = _nets_model.get_metrics(outputs,
                                          Variable(ex[-1]).cuda(),
                                          ex[-2])

        performance_dict['recall1'] += metrics[0]
        performance_dict['recall5'] += metrics[1]
        performance_dict['mrr'] += metrics[3]
        performance_dict['ieuc'] += metrics[4]
        performance_dict['count'] += outputs.data.size()[0]
        performance_dict['steps'] += 1.

        if d_idx % 1000 == 0 and d_idx > 0:
            print(d_idx)

    steps = performance_dict['steps']
    recall1 = performance_dict['recall1'] / steps
    recall5 = performance_dict['recall5'] / steps
    mrr = performance_dict['mrr'] / steps
    ieuc = performance_dict['ieuc'] / steps

    print('recall@1 %.4f' % recall1)
    print('recall@5 %.4f' % recall5)
    print('mrr      %.4f' % mrr)
    print('ieuc     %.4f' % ieuc)
    print('#events', performance_dict['count'])


def set_seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str,
                            default='./data/sample_data.csv')
    arg_parser.add_argument("--serialized_data_path", type=str,
                            default='./data/preprocess(test).pkl')
    arg_parser.add_argument("--model_path", type=str,
                            default='./data/nets_20180209.mdl')
    arg_parser.add_argument("--pretrained_dict_path", type=str,
                            default='./data/nets_k1k2_grid_dict.pkl')
    arg_parser.add_argument("--seed", type=int, default=3)
    args = arg_parser.parse_args()

    set_seed_all(args.seed)

    config = dataset.Config()
    config.test_path = args.input_path
    config.preprocess_save_path = args.serialized_data_path
    config.preprocess_load_path = args.serialized_data_path

    print('Loading test dataset')
    test_dataset = get_dataset(config, args.pretrained_dict_path)
    assert test_dataset is not None

    print('Loading NETS model')
    nets_model = get_model(test_dataset, args.model_path)

    print('Measuring NETS performance on test data')
    measure_performance(test_dataset, nets_model)
