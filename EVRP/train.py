##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 2


##########################################################################################
# Path Config

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for config


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from EVRPTrainer import EVRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'training_mode': True,
    'draw_picture': False,
    'customer_size': 50,
    'satellite_size': 5,
    'fe_car_capacity': 5,
    'se_car_capacity': 1,
    'normalization_coefficient' : 1,
    'lumda': 1,
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
    'satellite_size': 10,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [10800,10900,11000],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 11000,
    'train_episodes': 10*1000,
    'train_batch_size': 256,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 500,
        'img_save_interval': 500,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_2evrp.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/saved_2EVRP50_nnnn_model',  # directory path of pre-trained model and log files saved.
        'epoch': 10000,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train_2evrp_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    parser = argparse.ArgumentParser(description='VRP Training Script')
    
    parser.add_argument('--joint', action='store_true', help='Enable joint training mode')
    
    args = parser.parse_args()

    if args.joint:
        env_params['training_mode'] = False
    else:
        env_params['training_mode'] = True
    
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
