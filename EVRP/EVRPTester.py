
import torch

import os
from logging import getLogger

from EVRPEnv import EVRPEnv as Env
from EVRPModel import EVRPModel as Model

from utils.utils import *


class EVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_device('cuda')
        else:
            device = torch.device('cpu')
            torch.set_default_device('cpu')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):
        test_score_txt = "test_results_detailed.txt"
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()

        all_no_aug_scores = []
        all_aug_scores = []

        with open(test_score_txt, "w", encoding="utf-8") as f:
            f.write(f"{'Instance_ID':<15} | {'No-Aug Score':<15} | {'Aug Score':<15}\n")
            f.write("-" * 50 + "\n")

            with torch.no_grad():
                self.env.load_problems(batch_size, aug_factor)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

            # Return
            ###############################################
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

            for i in range(batch_size):
                no_aug_s = -max_pomo_reward[0, i].item()
                aug_s = -max_pomo_reward[:, i].max().item()
                
                all_no_aug_scores.append(no_aug_s)
                all_aug_scores.append(aug_s)
                
                f.write(f"ID {i:03d}          | {no_aug_s:<15.4f} | {aug_s:<15.4f}\n")

            avg_no_aug = sum(all_no_aug_scores) / len(all_no_aug_scores)
            avg_aug = sum(all_aug_scores) / len(all_aug_scores)
            
            summary = "\n" + "="*50 + "\n"
            summary += f"SUMMARY (Avg over {batch_size} instances):\n"
            summary += f"AVERAGE NO-AUG SCORE: {avg_no_aug:.4f}\n"
            summary += f"AVERAGE AUG SCORE:    {avg_aug:.4f}\n"
            summary += "="*50 + "\n"
            
            print(summary)
            f.write(summary)

        self.logger.info(f"Detailed results saved to {test_score_txt}")

        return no_aug_score.item(), aug_score.item()
