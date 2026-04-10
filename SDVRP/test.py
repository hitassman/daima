##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = 0
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import torch
from utils.utils import create_logger, copy_all_src

from SDVRPTester import SDVRPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'training_mode' : False,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': os.path.join(
            BASE_DIR,
            '../SDVRP/result/saved_SDVRP20_model'
        ),  # directory path of pre-trained model and log files saved.
        'epoch': 3000,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 8,
    'test_data_load': {
        'enable': True,
        'filename': 'A-n32-k5.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_sdvrp10',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    loaded_dict = torch.load(tester_params['test_data_load']['filename'])
    env_params['problem_size']=env_params['pomo_size']=loaded_dict['problem_size']

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
