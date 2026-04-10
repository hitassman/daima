##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = False
CUDA_DEVICE_NUM = 2


##########################################################################################
# Path Config

import os
import sys
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for config


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from EVRPTester import EVRPTester as Tester

##########################################################################################
# parameters

env_params = {
    'training_mode': False,
    'draw_picture': False,
    'normalization_coefficient' : 1,
    'lumda': 0,
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


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_2EVRP50_final_model',  # directory path of pre-trained model and log files saved.
        'epoch': 11000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 50,
    'test_batch_size': 64,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 500,
    'test_data_load': {
        'enable': True,
        'filename': './testcase/tmp.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_2evrp15',
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
    if len(sys.argv) != 2:
        print("Usage: python change.py <dat_filename>")
        print("Example: python change.py E-n51-k5-s02-04-17-46.dat")
        sys.exit(1) 
    
    # 获取输入文件名
    input_file = sys.argv[1]
        
    # 检查文件是否存在
    if not os.path.exists(input_file+'.pt'):
        print(f"Checking: {os.path.abspath(input_file + '.pt')}")
        print(f"Error: File '{input_file}.pt' not found!")
        sys.exit(1)
        
    # 处理文件
    try:
        dataset = torch.load(input_file+'.pt') 
        print("File processed successfully!")
        # 在这里添加你的后续处理代码...
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


    env_params['customer_size'] = dataset['customer_num']
    env_params['satellite_size'] = dataset['satellite_num']
    model_params['satellite_size'] = dataset['satellite_num']
    dataset['customer_demand'] = dataset['customer_demand'] / dataset['se_cars_capacity']
    torch.save(dataset,'./testcase/tmp.pt')
    env_params['fe_car_capacity'] = dataset['fe_cars_capacity'] / dataset['se_cars_capacity']
    env_params['se_car_capacity'] = 1
    env_params['normalization_coefficient'] = dataset['se_cars_capacity']
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
