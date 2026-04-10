
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from SDVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    problem_size: int = None
    selected_count: int = None
    is_rest: torch.Tensor = None
    rest_node: torch.Tensor = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
class SDVRPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.real_problem_size = self.problem_size
        self.pomo_size = env_params['pomo_size']
        self.training_mode = env_params['training_mode']
        self.fe_cars_capacity = 1
        self.tim = 0
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        # Dynamic-1
        ####################################
        self.selected_count = None
        self.is_rest = None
        # is_rest ==  0 : normal
        # is_rest ==  1 : rest selected this step (to be processed)
        # is_rest == -1 : rest already used (inactive)
        self.rest_node = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)
        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device, weights_only=False)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_depot_xy_origin = loaded_dict['depot_xy_origin']
        self.saved_node_xy_origin = loaded_dict['node_xy_origin']
        self.saved_index = 0
    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        if not self.FLAG__use_saved_problems:
            self.real_problem_size = self.problem_size - torch.randint(low=0, high=self.problem_size-1, size=(1,)).item()
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.real_problem_size)
            self.saved_depot_xy_origin = depot_xy
            self.saved_node_xy_origin = node_xy
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                self.saved_depot_xy_origin = augment_xy_data_by_8_fold(self.saved_depot_xy_origin)
                self.saved_node_xy_origin = augment_xy_data_by_8_fold(self.saved_node_xy_origin)
            else: 
                raise NotImplementedError
        
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.depot_node_xy_origin = torch.cat((self.saved_depot_xy_origin, self.saved_node_xy_origin), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.problem_size = self.real_problem_size
    def reset(self):
        self.selected_count = 0
        
        self.current_node = torch.zeros(size=(self.batch_size,self.pomo_size),dtype=torch.long)
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.is_rest = torch.zeros(size=(self.batch_size,self.pomo_size),dtype=torch.long)
        self.rest_node = torch.zeros(size=(self.batch_size,self.pomo_size),dtype=torch.long)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.full(size=(self.batch_size, self.pomo_size), fill_value=self.fe_cars_capacity, dtype=torch.float)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.real_problem_size+2))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.real_problem_size+2))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        zeros_cat = torch.zeros(self.batch_size,1)
        self.depot_node_demand = torch.cat((self.depot_node_demand,zeros_cat),dim=1)
        self.demand_list = self.depot_node_demand.unsqueeze(1).expand(self.batch_size,self.pomo_size,self.real_problem_size+2).clone()
        # shape: (batch,pomo,problem+1)
        reward = None
        done = False
        return self.reset_state, reward, done
    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        reward = None
        done = False
        return self.step_state, reward, done
    def step(self, selected):
        # Dynamic-1
        ####################################
        self.selected_count += 1
        rest_mask = (selected == self.real_problem_size + 1)
        
        rmask = (self.is_rest == 1)

        self.is_rest = torch.where(
            rest_mask,
            torch.full_like(self.is_rest, 1),
            self.is_rest
        )

        self.rest_node = torch.where(
            rmask,
            selected,
            self.rest_node
        )

        self.is_rest = torch.where(
            rmask,
            torch.full_like(self.is_rest, -1),
            self.is_rest
        )

        self.current_node = torch.where(rest_mask, self.current_node, selected)
        # shape: (batch, pomo)


        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)
        has_rest = (self.rest_node != 0)

        settle_mask = self.at_the_depot & has_rest
        if settle_mask.any():
            rest_demand = self.demand_list[
                self.BATCH_IDX, self.POMO_IDX, self.rest_node
            ]
 
            served = torch.minimum(self.load, rest_demand)


            self.demand_list[
                self.BATCH_IDX, self.POMO_IDX, self.rest_node
            ] = rest_demand - served

            self.load = self.load - served

        selected_demand = self.demand_list[self.BATCH_IDX, self.POMO_IDX, selected]
        selected_demand = torch.where(rmask,torch.zeros_like(selected_demand),selected_demand)
        # shape: (batch, pomo)
        mask = (self.load > selected_demand) & (~rmask)
        
        self.load[mask] -= selected_demand[mask]
        selected_demand[mask] = 0
        
        selected_demand[~mask] -= self.load[~mask]
        self.load[~mask] = 0
        self.demand_list[self.BATCH_IDX,self.POMO_IDX,selected] = selected_demand

        esp = 0.0000001
        mask = self.load >= -esp
        self.load = torch.max(self.load, torch.tensor(0))
        
        self.load[self.at_the_depot] = self.fe_cars_capacity # refill loaded at the depot
        self.is_rest[self.at_the_depot] = 0
        self.rest_node[self.at_the_depot] = 0
        self.visited_ninf_flag = torch.where(self.demand_list == 0, float('-inf'), 0)
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot
        self.visited_ninf_flag[:, :, self.real_problem_size+1] = float('-inf')
        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load < esp
        
        self.ninf_mask[:,:,1:] = torch.where(demand_too_large.unsqueeze(-1),float('-inf'),self.ninf_mask[:,:,1:])
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.is_rest = self.is_rest
        self.step_state.rest_node = self.rest_node

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done
    def _get_travel_distance(self):

        # print(self.reset_state.node_demand)
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy_origin[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        if not self.training_mode:
            flat_idx = torch.argmin(travel_distances.view(-1))
            

            best_batch = flat_idx // self.pomo_size
            best_pomo  = flat_idx % self.pomo_size

            best_dist = travel_distances[best_batch, best_pomo]
   
            best_route_idx = self.selected_node_list[best_batch, best_pomo] 

            vis_batch = 0
            all_xy = self.depot_node_xy_origin[vis_batch]   

            route = all_xy[best_route_idx]                  
            route_next = route.roll(dims=0, shifts=-1)

            depot_xy = all_xy[0]
            customer_xy = all_xy[1:]

            plt.figure(figsize=(8, 8))

            plt.scatter(
                customer_xy[:, 0].cpu(),
                customer_xy[:, 1].cpu(),
                c='blue',
                s=30,
                label='Customers',
                zorder=2
            )

            plt.scatter(
                depot_xy[0].cpu(),
                depot_xy[1].cpu(),
                c='red',
                s=120,
                marker='s',
                label='Depot',
                zorder=3
            )

            for p, q in zip(route, route_next):
                plt.plot(
                    [p[0].item(), q[0].item()],
                    [p[1].item(), q[1].item()],
                    color='orange',
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=1
                )

            plt.title(
                f"SDVRP Route (global best over batch×pomo)\n"
                f"Distance = {best_dist:.3f}"
            )
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.axis("equal")
            plt.grid(alpha=0.3)

            plt.savefig("output_sdvrp.png", dpi=200, bbox_inches='tight')
            plt.close()
        return travel_distances