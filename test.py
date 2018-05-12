import argparse
import dataset
from model import NETS
import numpy as np
import os
import pickle
import random
import torch


def get_dataset(cfg, trained_dict_path):
    print('Creating the test dataset pickle..', )
    nets_dictionary = pickle.load(open(trained_dict_path, 'rb'))
    test_set = dataset.NETSDataset(cfg, nets_dictionary)
    if len(test_set.test_data) == 0:
        print('no events')
        return None
    pickle.dump(test_set, open(cfg.preprocess_save_path, 'wb'))
    return test_set


def get_model(widx2vec, model_path, dvc, arg):
    model_dir, model_filename = os.path.split(model_path)
    checkpoint = torch.load(model_path,
                            map_location=None if 'cuda' == dvc.type else 'cpu')
    ckpt_config = checkpoint['config']
    ckpt_dict = vars(ckpt_config)
    ckpt_dict['yes_cuda'] = arg.yes_cuda  # overriding

    model = NETS(ckpt_config, widx2vec).to(dvc)
    model.config.checkpoint_dir = model_dir + '/'
    model.load_checkpoint(filename=model_filename[:-4])  # .pth
    # import pprint
    # pprint.PrettyPrinter().pprint(_model.config.__dict__)
    return model


def measure_performance(test_set, model, dvc, batch_size=1):
    performance_dict = dict()
    performance_dict['recall1'] = 0.
    performance_dict['recall5'] = 0.
    performance_dict['mrr'] = 0.
    performance_dict['ieuc'] = 0.
    performance_dict['count'] = 0
    performance_dict['steps'] = 0.

    model = model.eval()
    torch.set_grad_enabled(False)

    _, _, test_loader = test_set.get_dataloader(batch_size=batch_size)
    for d_idx, ex in enumerate(test_loader):
        labels = ex[-1].to(dvc)
        outputs, reps = model(*ex[:-1])
        metrics = model.get_metrics(outputs, labels, ex[-2].to(dvc))

        performance_dict['recall1'] += metrics[0]
        performance_dict['recall5'] += metrics[1]
        performance_dict['mrr'] += metrics[2]
        performance_dict['ieuc'] += metrics[3]
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
                            default='./data/preprocess_test.pkl')
    arg_parser.add_argument("--model_path", type=str,
                            default='./data/nets_180512_0.pth')
    arg_parser.add_argument("--trained_dict_path", type=str,
                            default='./data/preprocess_180504_dict.pkl')
    arg_parser.add_argument("--seed", type=int, default=3)
    arg_parser.add_argument('--yes_cuda', type=int, default=1)
    args = arg_parser.parse_args()

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')

    set_seed_all(args.seed)

    config = dataset.Config()
    config.test_path = args.input_path
    config.preprocess_save_path = args.serialized_data_path
    config.preprocess_load_path = args.serialized_data_path

    print('Loading test dataset..')
    test_dataset = get_dataset(config, args.trained_dict_path)
    assert test_dataset is not None

    print('Loading NETS model..')
    nets_model = get_model(test_dataset.widx2vec, args.model_path, device, args)

    print('\nMeasuring NETS performance on test data..')
    measure_performance(test_dataset, nets_model, device)
