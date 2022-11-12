import argparse
import dataset
from model import NESA, get_metrics
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


def get_model(widx2vec, model_path, dvc, idx2dur, arg):
    model_dir, model_filename = os.path.split(model_path)
    checkpoint = torch.load(model_path,
                            map_location=None if 'cuda' == dvc.type else 'cpu')
    ckpt_config = checkpoint['config']
    ckpt_dict = vars(ckpt_config)
    ckpt_dict['yes_cuda'] = arg.yes_cuda  # overriding

    model = \
        NESA(ckpt_config, widx2vec,
             idx2dur=idx2dur if ckpt_config.use_duration_scala > 0
             else None).to(dvc)
    model.config.checkpoint_dir = model_dir + '/'
    model.load_checkpoint(filename=model_filename[:-4])  # .pth
    # import pprint
    # pprint.PrettyPrinter().pprint(_model.config.__dict__)
    return model, ckpt_config


def measure_performance(test_set, model, conf, dvc, batch_size=1):
    model = model.eval()

    performance_dict = dict()
    performance_dict['recall1'] = 0.
    performance_dict['recall5'] = 0.
    performance_dict['mrr'] = 0.
    performance_dict['ieuc'] = 0.

    _, _, test_loader = test_set.get_dataloader(batch_size=batch_size)
    with torch.inference_mode():
        for d_idx, ex in enumerate(test_loader):
            labels = ex[-1].to(dvc)
            outputs = model(*ex[:-1])
            metrics = get_metrics(outputs, labels, model.n_day_slots,
                                  model.n_classes,
                                  ex_targets=ex[-2].to(device)
                                  if conf.ex_pre_events > 0 else None)

            performance_dict['recall1'] += metrics[0]
            performance_dict['recall5'] += metrics[1]
            performance_dict['mrr'] += metrics[2]
            performance_dict['ieuc'] += metrics[3]

            if d_idx % 1000 == 0 and d_idx > 0:
                print(d_idx)

    n_samples = len(test_loader.dataset)
    recall1 = performance_dict['recall1'] / n_samples
    recall5 = performance_dict['recall5'] / n_samples
    mrr = performance_dict['mrr'] / n_samples
    ieuc = performance_dict['ieuc'] / n_samples

    print('recall@1 %.4f' % recall1)
    print('recall@5 %.4f' % recall5)
    print('mrr      %.4f' % mrr)
    print('ieuc     %.4f' % ieuc)
    print('#events', n_samples)


def set_seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str,
                            default='./data/sample_data.csv')
    arg_parser.add_argument("--serialized_data_path", type=str,
                            default='./data/preprocess_test.pkl')
    arg_parser.add_argument("--model_path", type=str,
                            default='./data/nesa_180522_0.pth')
    arg_parser.add_argument("--trained_dict_path", type=str,
                            default='./data/dataset_180522_dict.pkl')
    arg_parser.add_argument("--seed", type=int, default=3)
    arg_parser.add_argument('--yes_cuda', type=int, default=1)
    args = arg_parser.parse_args()

    use_cuda = args.yes_cuda > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('CUDA device_count {0}'.format(torch.cuda.device_count())
          if use_cuda else 'CPU')
    print(torch.__version__)

    set_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    config = dataset.Config()
    config.test_path = args.input_path
    config.preprocess_save_path = args.serialized_data_path
    config.preprocess_load_path = args.serialized_data_path

    print('Loading test dataset..')
    test_dataset = get_dataset(config, args.trained_dict_path)
    assert test_dataset is not None

    print('Loading NESA model..')
    nesa_model, nesa_conf = get_model(test_dataset.widx2vec, args.model_path,
                                      device, test_dataset.idx2dur, args)

    print('\nMeasuring NESA performance on test data..')
    measure_performance(test_dataset, nesa_model, nesa_conf, device)
