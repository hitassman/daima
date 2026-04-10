
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from EVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

##########################################################################################
# Path Config

import os
import sys
import multiprocessing as mp
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for config

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    
    satellite_xy: torch.Tensor = None
    # shape: (batch, satellite, 2)
    
    customer_xy: torch.Tensor = None
    # shape: (batch, customer, 2)

    customer_demand: torch.Tensor = None
    # shape: (batch, customer)

    storage_price_for_satellite: torch.Tensor = None
    # shape: (batch, satellite)

    depot_xy_origin: torch.Tensor = None
    # shape: (batch, 1, 2)
    
    satellite_xy_origin: torch.Tensor = None
    # shape: (batch, satellite, 2)
    
    customer_xy_origin: torch.Tensor = None
    # shape: (batch, customer, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    satellite_size: int = None
    customer_size: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    current_xy: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, 1+satellite+customer)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    current_satellite: torch.Tensor = None
    # shape: (batch, pomo)
    count_satellite:torch.Tensor = None
    # shape: (batch, pomo, satellite+1)


class EVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.training_mode = env_params['training_mode']
        self.draw_picture = env_params['draw_picture']
        self.satellite_size = env_params['satellite_size']
        self.real_satellite_size = self.satellite_size
        self.customer_size = env_params['customer_size']
        self.real_customer_size = self.customer_size
        self.pomo_size = self.real_satellite_size * self.real_customer_size
        self.fe_car_capacity = env_params['fe_car_capacity']
        self.se_car_capacity = env_params['se_car_capacity']
        self.normalization_coefficient = env_params['normalization_coefficient']
        self.lumda = env_params['lumda']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_satellite_xy = None
        self.saved_customer_xy = None
        self.saved_depot_xy_origin = None
        self.saved_satellite_xy_origin = None
        self.saved_customer_xy_origin = None
        self.saved_customer_demand = None
        self.saved_storage_price_for_satellite = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, satellite+customer+1, 2)
        self.depot_node_xy_origin = None
        # shape: (batch, satellite+customer+1, 2)
        self.depot_node_demand = None
        # shape: (batch, satellite+customer+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.step_counter = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_satellite = None
        # shape: (batch, pomo)
        self.at_the_customer = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, 1+satellite+customer)
        self.ninf_mask = None
        # shape: (batch, pomo, 1+satellite+customer)
        self.finished = None
        # shape: (batch, pomo)
        self.count_satellite = None
        # shape: (batch, pomo, satellite+1)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        keys_list = list(loaded_dict.keys())
        # print('keys_list: ',keys_list)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_satellite_xy = loaded_dict['satellite_xy']
        self.saved_customer_xy = loaded_dict['customer_xy']
        self.saved_customer_demand = loaded_dict['customer_demand']
        self.saved_storage_price_for_satellite = loaded_dict['storage_price_for_satellite']
        self.saved_depot_xy_origin = loaded_dict['depot_xy_origin']
        self.saved_satellite_xy_origin = loaded_dict['satellite_xy_origin']
        self.saved_customer_xy_origin = loaded_dict['customer_xy_origin']

        self.saved_index = 0
        _,self.real_customer_size,_ = self.saved_customer_xy.size()
        _,self.real_satellite_size,_ = self.saved_satellite_xy.size()


    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            self.real_customer_size = self.satellite_size
            self.real_satellite_size = self.customer_size
            depot_xy, satellite_xy, customer_xy, customer_demand, storage_price_for_satellite = \
                get_random_problems(batch_size, self.real_customer_size, self.real_satellite_size)
            depot_xy_origin = depot_xy
            satellite_xy_origin = satellite_xy
            customer_xy_origin = customer_xy
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            satellite_xy = self.saved_satellite_xy[self.saved_index:self.saved_index+batch_size]
            customer_xy = self.saved_customer_xy[self.saved_index:self.saved_index+batch_size]
            customer_demand = self.saved_customer_demand[self.saved_index:self.saved_index+batch_size]
            storage_price_for_satellite = self.saved_storage_price_for_satellite[self.saved_index:self.saved_index+batch_size]
            depot_xy_origin = self.saved_depot_xy_origin[self.saved_index:self.saved_index+batch_size]
            satellite_xy_origin = self.saved_satellite_xy_origin[self.saved_index:self.saved_index+batch_size]
            customer_xy_origin = self.saved_customer_xy_origin[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size
        self.pomo_size = self.real_satellite_size * self.real_customer_size
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                satellite_xy = augment_xy_data_by_8_fold(satellite_xy)
                customer_xy = augment_xy_data_by_8_fold(customer_xy)
                depot_xy_origin = augment_xy_data_by_8_fold(depot_xy_origin)
                satellite_xy_origin = augment_xy_data_by_8_fold(satellite_xy_origin)
                customer_xy_origin = augment_xy_data_by_8_fold(customer_xy_origin)
                customer_demand = customer_demand.repeat(8, 1)
                storage_price_for_satellite = storage_price_for_satellite.repeat(8,1)
            else:
                raise NotImplementedError
        self.depot_node_xy = torch.cat((depot_xy, satellite_xy), dim=1)
        self.depot_node_xy = torch.cat((self.depot_node_xy,customer_xy),dim=1)
        # shape: (batch, satellite+customer+1, 2)
        self.depot_node_xy_origin = torch.cat((depot_xy_origin, satellite_xy_origin), dim=1)
        self.depot_node_xy_origin = torch.cat((self.depot_node_xy_origin,customer_xy_origin),dim=1)

        self.depot_xy_expanded = depot_xy_origin[:, 0, :].unsqueeze(1).expand(-1, satellite_xy_origin.size(1), -1)

        # shape: (batch, satellite+customer+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1+self.real_satellite_size))
        # shape: (batch, 1+satellite)
        self.depot_node_demand = torch.cat((depot_demand, customer_demand), dim=1)
        # shape: (batch, 1+satellite+customer)
        self.demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.satellite_xy = satellite_xy
        self.reset_state.customer_xy = customer_xy
        self.reset_state.depot_xy_origin = depot_xy_origin
        self.reset_state.satellite_xy_origin = satellite_xy_origin
        self.reset_state.customer_xy_origin = customer_xy_origin
        self.reset_state.customer_demand = customer_demand
        self.reset_state.storage_price_for_satellite = storage_price_for_satellite

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.satellite_size = self.real_satellite_size
        self.step_state.customer_size = self.real_customer_size

    def reset(self):
        self.selected_count = 0
        self.step_counter = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        self.current_node = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.current_satellite = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.count_satellite = torch.zeros(size=(self.batch_size, self.pomo_size, self.real_satellite_size))
        # shape: (batch, pomo, satellite)
        self.selected_node_list = [
            [torch.tensor([], dtype=torch.long) for _ in range(self.pomo_size)] 
            for _ in range(self.batch_size)
        ]
        # shape: (batch, pomo, 0~)

        self.at_the_satellite = torch.ones(size=(self.batch_size,self.pomo_size), dtype=torch.bool)
        # shape: (batch,pomo)
        self.at_the_customer = torch.zeros(size=(self.batch_size,self.pomo_size), dtype=torch.bool)
        # shape: (batch,pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)) * self.se_car_capacity
        # shape: (batch, pomo)
        
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.real_satellite_size+self.real_customer_size+1))
        # shape: (batch, pomo, satellite+customer+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.real_satellite_size+self.real_customer_size+1))
        # shape: (batch, pomo, satellite+customer+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.money = torch.zeros(size=(self.batch_size,self.pomo_size))
        self.step_state.current_xy = self.depot_node_xy_origin.gather(1,self.current_node.unsqueeze(-1).expand(-1, -1, 2))

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_satellite = self.current_satellite
        self.step_state.count_satellite = self.count_satellite

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.at_the_satellite = (selected > 0) & (selected <= self.real_satellite_size)
        self.at_the_customer = (selected > self.real_satellite_size)
        self.selected_count += 1

        selected_xy = self.depot_node_xy_origin.gather(1,selected.unsqueeze(-1).expand(-1, -1, 2))
        current_xy = self.step_state.current_xy
        current_satellite_xy = self.depot_node_xy_origin.gather(1,self.current_satellite.unsqueeze(-1).expand(-1, -1, 2))
      
        dist_to_selected = torch.norm(current_xy - selected_xy, dim=2)  # (batch, pomo)
        dist_to_satellite = torch.norm(current_xy - current_satellite_xy, dim=2)  # (batch, pomo)
        self.money += torch.where(self.at_the_satellite,dist_to_satellite,dist_to_selected)
                
        gmask = self.at_the_satellite
        self.current_satellite[gmask] = selected[gmask] 
        self.current_node = selected
        
        # Dynamic-2
        ####################################
        gathering_index = selected[:, :, None]

        # shape: (batch, pomo, 1)
        selected_demand = self.demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        indices = self.current_satellite.unsqueeze(-1)
        indices = indices - 1
        updates = selected_demand.unsqueeze(-1)
        self.count_satellite.scatter_add_(2, indices, updates)

        self.load -= selected_demand
        storage_price_expanded = self.reset_state.storage_price_for_satellite.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.real_satellite_size)
       
        tmp_current_satellite = torch.max(self.current_satellite - 1, torch.tensor(0))

        selected_storage_price = torch.gather(storage_price_expanded, 2, tmp_current_satellite.unsqueeze(-1)).squeeze(-1)

        self.money += selected_demand * selected_storage_price * self.normalization_coefficient

        self.load[self.at_the_satellite] = 1 # refill loaded at the depot
        self.load.relu_()

        mask = (selected > self.real_satellite_size)
        self.visited_ninf_flag[self.BATCH_IDX[mask], self.POMO_IDX[mask], selected[mask]] = float('-inf')

        # shape: (batch, pomo, problem+1)

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < self.demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        self.ninf_mask[:,:,0] = float('-inf')
        self.ninf_mask[:,:,1 : 1+self.real_satellite_size] = torch.where(self.at_the_satellite.unsqueeze(-1), float('-inf'), self.ninf_mask[:,:,1 : 1+self.real_satellite_size])
        
        finished = (self.visited_ninf_flag[:,:,1+self.real_satellite_size : 1+self.real_satellite_size+self.real_customer_size] == float('-inf')).all(dim=2) & (self.at_the_satellite)
        # shape: (batch, pomo)
        newly_finished = finished & (~self.finished)
        gmask = newly_finished
        

        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)
        self.step_counter[~self.finished] += 1

        # do not mask depot for finished episode.
        self.ninf_mask[:, : ,None][self.finished] = float('-inf')
        self.ninf_mask[self.BATCH_IDX[finished], self.POMO_IDX[finished], self.current_node[finished]] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_xy = selected_xy

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):

        if not self.training_mode:
            _,satellite_size,_ = self.reset_state.satellite_xy.size()
            self.satellite_demand = self.count_satellite
            ceil_div = torch.floor(self.count_satellite / self.fe_car_capacity) 
            self.satellite_demand -= ceil_div * self.fe_car_capacity
            sdvrpset={}
            sdvrpset['depot_xy_origin'] = self.reset_state.depot_xy_origin.repeat(1,self.pomo_size,1,1).view(self.batch_size*self.pomo_size,1,2)
            sdvrpset['node_xy_origin'] = self.reset_state.satellite_xy_origin.repeat(1,self.pomo_size,1,1).view(self.batch_size*self.pomo_size,satellite_size,2)
            sdvrpset['depot_xy'] = self.reset_state.depot_xy.repeat(1,self.pomo_size,1,1).view(self.batch_size*self.pomo_size,1,2)
            sdvrpset['node_xy'] = self.reset_state.satellite_xy.repeat(1,self.pomo_size,1,1).view(self.batch_size*self.pomo_size,satellite_size,2)
            sdvrpset['node_demand'] = (self.satellite_demand/self.fe_car_capacity).view(self.batch_size*self.pomo_size,satellite_size)
            torch.save(sdvrpset, './testcase/tmp.pt')
            env_params = {
                'problem_size': satellite_size,
                'pomo_size': satellite_size,
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
            tester_params = {
                'use_cuda': 0,
                'cuda_device_num': 2,
                'model_load': {
                    'path': '../SDVRP/result/saved_SDVRP10_model',  # directory path of pre-trained model and log files saved.
                    'epoch': 3000,  # epoch version of pre-trained model to laod.
                },
                'test_episodes': self.batch_size*self.pomo_size,
                'test_batch_size': 256,
                'augmentation_enable': False,
                'aug_factor': 8,
                'aug_batch_size': 32,
                'test_data_load': {
                    'enable': True,
                    'filename': './testcase/tmp.pt'
                },
            }
            import os
            import sys
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, "../SDVRP")  # for SDVRP

            from SDVRPTester import SDVRPTester as Tester
            tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)
            tester.run()

            if ['augmentation_enable'] == True:
                results = tester.print_aug_reward()
            else:
                results = tester.print_aug_reward()

            cost_of_sdvrp = torch.tensor(results).view(self.batch_size,self.pomo_size)
        else:
            ceil_div = torch.ceil(self.count_satellite / self.fe_car_capacity) 

        satellite_xy_origin = self.reset_state.satellite_xy_origin

        distances = torch.norm(satellite_xy_origin - self.depot_xy_expanded, dim=2)  # (batch, satellites)
        distances = distances * 2 
        distances = distances.unsqueeze(1).expand(-1, self.pomo_size, -1)

        ans_update = (distances * ceil_div).sum(dim=2)  # (batch, pomo)
        if not self.training_mode:
            ans = self.money + ans_update + cost_of_sdvrp
        else:
            ans = self.money + self.lumda * ans_update
        
        
        if self.draw_picture:
            # draw a picture
            min_index = ans.argmin()
            row, col = divmod(min_index.item(), ans.size(1))
            x = self.reset_state.depot_xy[0][0][0]
            y = self.reset_state.depot_xy[0][0][1]
            X = []
            Y = []
            X.append(x.item())
            Y.append(y.item())
            plt.scatter(x,y,color='green',s=100,label='Depot')
            x = [p[0].item() for p in self.reset_state.customer_xy[0]]
            y = [p[1].item() for p in self.reset_state.customer_xy[0]]
            
            plt.scatter(x,y,color='blue',s=100,label='Customer')
            x = [p[0].item() for p in self.reset_state.satellite_xy[0]]
            y = [p[1].item() for p in self.reset_state.satellite_xy[0]]
            X = X + x
            Y = Y + y
            plt.scatter(x,y,color='yellow',s=100,label='Satellite')

            x = [p[0].item() for p in self.reset_state.customer_xy[0]]
            y = [p[1].item() for p in self.reset_state.customer_xy[0]]
            X = X + x
            Y = Y + y

            ordered_seq = self.selected_node_list[row][col]
            ordered_seq = ordered_seq[1:-1]
            print("ordered_seq ",ordered_seq)
            rolled_seq = ordered_seq.roll(dims=0,shifts=-1)
            tim = 0
            for start,end in zip(ordered_seq,rolled_seq):
                plt.plot([X[start.item()],X[end.item()]],[Y[start.item()],Y[end.item()]],color='red', label='Edges' if tim == 0 else "")
                tim += 1

            plt.title('Real 2E-VRP DSP version')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')

            plt.legend()

            # plt.show()
            plt.savefig('output.png')

        return ans