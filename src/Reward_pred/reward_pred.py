import os
import sys
import numpy as np
import random
import pprint
import yaml
import gym
import math
import pathlib
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .Siamese_network import Siam__FC_linear_nn, Siam__conv_nn
from .Siamese_network import weight_init

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

# TODO: Maybe insert those in arguments?
PRINT_BATCHES = 50

SIAMESE_FC_ARCH = [[64, 64], [256, 256], [1024, 256, 64]]
#SIAMESE_FC_ARCH = [[1024,1024], [1024,256,64], [1024,256]]

class Predictor:  # TODO:Change all defaults to None for loading option?
    def __init__(self, num_of_models=3, model_load_path=None, device=None, threshold=0, show_trajectory=None,
                 obs_space=None, save_results_path=None, target_update_interval=2, update_tau=0.01,
                 log_train_data_interval=1, wandb_project=None, r_norm_factor=0, action_prediction_space=None):
        '''
            r_norm_factor: if not 0, normalize reward for siamese training:
                            R = (R - rnf / 2) / (rnf / s)
        '''
        self.Siamese_models = []
        self._num_of_models = num_of_models
        # self._input_layer_size = input_fc_layer_size
        self._threshold = threshold
        self._show_trajectory = show_trajectory
        self._itr = 0
        self._target_update_interval = target_update_interval
        self._log_train_data_interval = log_train_data_interval
        self._update_tau = update_tau
        self._predict_with_action = action_prediction_space is not None

        if self._predict_with_action:
            if isinstance(action_prediction_space, gym.spaces.Space):
                self._action_space = np.asarray(action_prediction_space.sample()).size
            elif type(action_prediction_space) == type((1,)):  # TODO: Need to check this for other envs other than random rooms
                self._action_space = action_prediction_space
            else:
                print("Action space is not valid")
                exit(1)
        else:
            self._action_space = None

        if type(obs_space) == gym.spaces.Box:
            self._obs_space_shape = obs_space.shape
        elif type(obs_space) == type((1,)):
            self._obs_space_shape = obs_space
        else:
            print("Obs space is not valid")
            exit(1)
        siamese_type = "linear" if len(self._obs_space_shape) == 1 else "conv"

        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.Siamese_models = [0] * self._num_of_models
        if model_load_path:
            yml_path = ""
            for subdir, dirs, files in os.walk(model_load_path):
                for file in files:
                    if not file.endswith(".yml") and not file.endswith(".yaml"):
                        filepath = subdir + os.sep + file
                        arch_index = int(file[-4:-3])
                        if siamese_type == "conv":
                            self.Siamese_models[arch_index] = Siam__conv_nn(self._obs_space_shape, 1, SIAMESE_FC_ARCH[arch_index],
                                                                            action_space=self._action_space)
                        else:
                            input_layer_size = self._obs_space_shape[0]
                            self.Siamese_models[arch_index] = Siam__FC_linear_nn(input_layer_size, SIAMESE_FC_ARCH[arch_index], 1,
                                                                                 action_space=self._action_space)
                        loaded_state_dict = torch.load(filepath)
                        self.Siamese_models[arch_index].load_state_dict(loaded_state_dict)
                        self.Siamese_models[arch_index] = self.Siamese_models[arch_index].to(self.device)
                    else:
                        yml_path = subdir + os.sep + file

            if yml_path == "" or not os.path.isfile(yml_path):
                print ("Did not found any yaml config path for this RP load path")
                exit(1)
            else:
                with open (yml_path, "r") as F:
                    self._update_dict(yaml.load(F))
            print("Following are the loaded model configuration paramaters")
            pprint.pprint(self.__dict__, depth=1, width=60)

        else:
            # Replay_buffer is used only to train
            self._train_counter = 0
            self.replay_buffer = [[], [], []]
            self._save_results_path = save_results_path
            self.train_results = []

            if siamese_type == "conv":
                for i in range(self._num_of_models):
                    self.Siamese_models[i] = (Siam__conv_nn(self._obs_space_shape, 1, SIAMESE_FC_ARCH[i],
                                                             action_space=self._action_space))  # TODO: Is 1 need to be argument?
                    self.Siamese_models[i].apply(weight_init)  # apply weight init - check if it works for the conv_nn model
                    self.Siamese_models[i] = self.Siamese_models[i].to(self.device)
            else:
                input_layer_size = self._obs_space_shape[0]
                for i in range(self._num_of_models):
                    self.Siamese_models[i] = (Siam__FC_linear_nn(input_layer_size, SIAMESE_FC_ARCH[i], 1,
                                                                  action_space=self._action_space))
                    self.Siamese_models[i].apply(weight_init)  # apply weight init
                    self.Siamese_models[i] = self.Siamese_models[i].to(self.device)

            # Define optimizer for every Siamese arch
            self.optimizers = []
            for i in range(self._num_of_models):
                self.optimizers.append(optim.Adam(self.Siamese_models[i].parameters(), lr=0.001, weight_decay=0.001))

        if wandb_project:
            for model in self.Siamese_models:
                wandb.watch(model)
        self.wandb_project = wandb_project

        self. _inner_dict = {key:val for key, val in self.__dict__.items() if key.startswith("_")}

        self.Target_Siamese_models = []
        for i in range(self._num_of_models):
            if siamese_type == "conv":
                self.Target_Siamese_models.append(Siam__conv_nn(self._obs_space_shape, 1, SIAMESE_FC_ARCH[i],
                              action_space=self._action_space))
            else:
                self.Target_Siamese_models.append(Siam__FC_linear_nn(input_layer_size, SIAMESE_FC_ARCH[i], 1,
                                                              action_space=self._action_space))
            self.Target_Siamese_models[i].load_state_dict(copy.deepcopy(self.Siamese_models[i].state_dict()))
            self.Target_Siamese_models[i] = self.Target_Siamese_models[i].to(self.device)

    # Temp solution
        self.all_losses = []

        self.r_norm_factor = r_norm_factor
    
    def get_tensorBoard_Writer(self):
        return self.tensorboard_writer

    def _restart_lr_scheduler(self, epoch_len):
        self.schedulers = []
        for i in range(self._num_of_models):
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[i], epoch_len, eta_min = 0.00001)) #TODO:Change 1000 to argument

    # -------------------------- General functions --------------------------

    def _update_dict(self, config_dict):
        for key, item in config_dict.items():
            if not key in self.__dict__.keys() or not self.__dict__[key]:
                self.__dict__[key] = item

    def save_models(self, target_dir, model_name):
        for i in range(self._num_of_models):
            tmp_model_name = "Siamese_model_{}_{}".format(model_name, str(i)) + ".pt"
            # target_tmp_model_name = "Target_Siamese_model_{}_{}".format(model_name,str(i)) + ".pt"
            model_path = os.path.join(target_dir, tmp_model_name)
            # target_model_path = os.path.join(target_dir, target_tmp_model_name)
            torch.save(self.Siamese_models[i].state_dict(), model_path)
            # torch.save(self.Target_Siamese_models[i].state_dict(), target_model_path)
            # print("Saved Siamese model under name: {}".format(model_path))

        config_file_path = os.path.join(target_dir, model_name  + "_config_file.yml")
        with open(config_file_path, "w+") as F:
            yaml.dump(self._inner_dict, F)
            # print("Saved Siamese model config file under name: {}".format(config_file_path))
            
    # -------------------------- Train functions --------------------------

    def one_batch_train_temp_for_subproc(self, replay_buffer, device, batch_size=64):
        self._itr += 1
        loss_func = torch.nn.MSELoss()
        L1_loss_func = torch.nn.L1Loss()
        current_loss = []

        for model in range(self._num_of_models):
            batch_tensor, rewards_tensor, action_tensor = replay_buffer.get_one_batch(batch_size, device)
            rewards_tensor = self.norm_reward(rewards_tensor, 'in')

            # Calculate network outputs
            temp_predicted_rewards = self.Siamese_models[model](batch_tensor, action_tensor)
            temp_predicted_rewards = self.norm_reward(temp_predicted_rewards, 'out')

            # Train Siamese network:
            self.optimizers[model].zero_grad()

            # - Compute loss
            loss = loss_func(temp_predicted_rewards, rewards_tensor)

            if (self._itr % self._log_train_data_interval == 0):
                L1_loss = L1_loss_func(temp_predicted_rewards, rewards_tensor)
                if (self.wandb_project):
                    wandb.log({"RP_Train/Siamese_model_{}_MSE_loss".format(model): loss})
                    wandb.log({"RP_Train/Siamese_model_{}_L1_loss".format(model): L1_loss})
                L1_loss.detach().cpu()

            current_loss.append("{:.5f}".format(loss))

            # - Optimize the model
            loss.backward()
            self.optimizers[model].step()

        self.all_losses.append(current_loss)

        if (self._itr % self._target_update_interval == 0):
            for i in range(self._num_of_models):
                update_state_dict(self.Target_Siamese_models[i], self.Siamese_models[i].state_dict(), self._update_tau)

        if (len(self.all_losses) >= 50):
            print_results(self.all_losses, self._num_of_models, is_train=True)
            self.all_losses = []

    # TODO: This is a functin for off_line train only (?)
    def train(self, windows_list, actions_list, acc_rewards_list, test_obs, test_actions, test_rewards, device, results_dir, model_name = "", num_of_epochs = 6, batch_size = 32, wandb_project = None,  win_size = 1):
        index_of_data = list(range(windows_list.shape[0]))
        epoch_len = math.floor(len(windows_list) / float(batch_size))

        tb_counter = 0
        random.shuffle(index_of_data)

        index_of_test_set = list(range(len(test_obs)))

        print ("train set size = {}".format(len(windows_list)))

        loss_func = torch.nn.MSELoss()
        L1_loss_func = torch.nn.L1Loss()

        for epoch in range(num_of_epochs):

            print("\n\n############# epoch {} ###############\n\n".format(epoch))
            random.shuffle(index_of_data)
            batch_counter = 0
            print_counter = 0
            print_results_list = []
            windows_batch_tensors = []
            rewards_batch_tensors = []
            actions_batch_tensors = []

            self._restart_lr_scheduler(epoch_len)

            # Go over all data
            for index in index_of_data:
                windows_batch_tensors.append(torch.FloatTensor(windows_list[index]))
                actions_batch_tensors.append(torch.FloatTensor(actions_list[index]))
                rewards_batch_tensors.append(torch.FloatTensor(acc_rewards_list[index]))
                batch_counter += 1

                if batch_counter >= batch_size:
                    tb_counter += 1
                    #print ("batch number {}".format(tb_counter))
                    print_counter += 1
                    current_results = []
                    batch_tensor = torch.stack(windows_batch_tensors).to(device)
                    actions_tensor = torch.stack(actions_batch_tensors).to(device)
                    rewards_tensor = torch.stack(rewards_batch_tensors).to(device)
                    # Calculate network outputs
                    temp_predicted_rewards = []
                    for i in range(self._num_of_models):
                        temp_predicted_rewards.append(self.Siamese_models[i](batch_tensor, actions_tensor))

                    # Train each Siamese network
                   
                    for i in range(self._num_of_models):
                        self.optimizers[i].zero_grad()

                        # Compute loss
                        loss = loss_func(temp_predicted_rewards[i], rewards_tensor)

                        L1_loss = L1_loss_func(temp_predicted_rewards[i], rewards_tensor)
                        if (self.wandb_project):
                            wandb.log({f"Siamese_model_{i}/MSE_loss": loss})
                            wandb.log({f"Siamese_model_{i}/L1_loss": L1_loss})
                        L1_loss.detach().cpu()

                        current_results.append("{:.5f}".format(loss))

                        # Optimize the model
                        loss.backward()
                        self.optimizers[i].step()
                        self.schedulers[i].step()

                    #Resset batch data
                    batch_counter = 0
                    windows_batch_tensors = []
                    rewards_batch_tensors = []
                    actions_batch_tensors = []

                    #Print train results
                    print_results_list.append (current_results)
                    if (print_counter >= 20):
                        print_results(print_results_list, self._num_of_models,  is_train = True, wandb_usage = wandb_project, win_size = win_size)
                        print_counter = 0
                        print_results_list = []
                    
                    if (tb_counter % 100  == 0): #TODO:Change this to argument
                        #Test set part
                        random.shuffle(index_of_test_set)
                        batch_counter = 0
                        test_losses = []
                        for index in index_of_test_set:
                            windows_batch_tensors.append(torch.FloatTensor(test_obs[index]))
                            actions_batch_tensors.append(torch.FloatTensor(test_actions[index]))
                            rewards_batch_tensors.append(torch.FloatTensor(test_rewards[index]))
                            batch_counter += 1
            
                            if batch_counter >= batch_size:
                                batch_tensor = torch.stack(windows_batch_tensors).to(device)
                                actions_tensor = torch.stack(actions_batch_tensors).to(device)
                                rewards_tensor = torch.stack(rewards_batch_tensors).to(device)
                                # Calculate network outputs
                                temp_predicted_rewards = []
                                for i in range(self._num_of_models):
                                    temp_predicted_rewards.append(self.Siamese_models[i](batch_tensor, actions_tensor))
            
                                curr_loss = []
                                for i in range(self._num_of_models):
                                    # Compute loss
                                    loss = loss_func(temp_predicted_rewards[i], rewards_tensor)
                                    L1_loss = L1_loss_func(temp_predicted_rewards[i], rewards_tensor)
                                    if (self.wandb_project):
                                        wandb.log({"Siamese_model_{}/test_MSE_loss".format(i): loss})
                                        wandb.log({"Siamese_model_{}/test_L1_loss".format(i): L1_loss})
                                    L1_loss.detach().cpu()
                                    curr_loss.append("{:.5f}".format(loss))
                                
                                test_losses.append (curr_loss)
                                curr_loss = []
                                
                                batch_counter = 0
                                windows_batch_tensors = []
                                rewards_batch_tensors = []
                                actions_batch_tensors = []
                       
                        #Print test results
                        print("\n\n ------------ test set results: ----------- \n\n")
                        print_results(test_losses, self._num_of_models,  is_train = False, wandb_usage = wandb_project, win_size = win_size)
                        print("\n\n ------------ test set results: ----------- \n\n")
                        batch_counter = 0
                        windows_batch_tensors = []
                        rewards_batch_tensors = []
                        actions_batch_tensors = []
                                
            #restart lr-scheduler
            self._restart_lr_scheduler(epoch_len)
                  
            #Save current Siamese models
            model_dir = pathlib.Path(os.path.join(results_dir, "Siamese_models", "epoch_{}".format(epoch)))
            model_dir.mkdir(parents=True, exist_ok=True)
            self.save_models(model_dir, model_name)

    # -------------------------- Predict functions --------------------------
    
    def predict(self, state, action=None):
        state_input = (torch.FloatTensor(state).to(self.device))
        if action is not None:
            action_input = (torch.FloatTensor(action).to(self.device))
        else:
            action_input = None

        # Get predicted reward for each env (for each model)
        rewards_array = []
        rewards_dict = {}
        for i in range(self._num_of_models):
            with torch.no_grad():
                R = self.Target_Siamese_models[i](state_input, action_input)
            R = self.norm_reward(R, 'out')
            rewards_array.append(R)
            for e, r in enumerate(R.cpu().detach()):
                rewards_dict["env_{}/Siamese_{}_reward".format(e, i)] = float(r)
        if self.wandb_project:
            wandb.log(rewards_dict)

        rewards_array = torch.stack(rewards_array).to("cpu")
        rewards_ndarray = rewards_array.detach().numpy()      
        return_rewards = np.average(rewards_ndarray, axis=0) # Get avergae score of all models
        # return_rewards = np.max(rewards_ndarray, axis=0) #Get max score of all models
        var_reward = np.var(rewards_ndarray, axis=0) # Get variance of scores from all models
        return_rewards = return_rewards.reshape((return_rewards.shape[0],))
        var_reward = var_reward.reshape((var_reward.shape[0],))

        return return_rewards, var_reward

    def norm_reward(self, R, mod):
        # R: can be scalar / tensor of rewards
        # mod: 'in' or 'out'
        rnf = self.r_norm_factor 
        if  rnf == 0:
            return R
        elif mod == 'in':
            R = (R - (rnf / 2)) * (2 / rnf)
        elif mod == 'out':
            R = (R * rnf) + (rnf / 2)
        else:
            assert False, "mod is {} illegal - not in or out".format(mod)
        return R


def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """
    Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    # if strip_ddp:
    #    state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)


# ---------------------- 3) Train printing functions ------------------------------------

def parseResults(results_list, num_of_models):
    import statistics
    sum_of_results = []

    for i in range(num_of_models):
        temp_results = [float(result[i]) for result in results_list]
        max_model_result = max(temp_results)
        min_model_result = min(temp_results)
        mean_model_result = float(sum(temp_results)) / float(len(temp_results))
        median_model_result = statistics.median(temp_results)
        sum_of_results.append (max_model_result)
        sum_of_results.append(min_model_result)
        sum_of_results.append(mean_model_result)
        sum_of_results.append(median_model_result)

    return sum_of_results


def print_results(results, num_of_models, is_train = True, verbose = True, wandb_usage = None, win_size = 1):
    train_text = "Train" if is_train else "Test"
    results_sumary = parseResults(results, num_of_models)
    if verbose: print("Siamese models mean loss values are:\n")
    for i in range(num_of_models):
     #Log to wandb the avg test result:
        
        start_index = 4 * i
        
        if (wandb_usage):
            wandb.log({"Model {} - mean {} loss".format(str(i), train_text) : results_sumary[start_index + 2]})
            wandb.log({"Model {} - mean {} normalized loss".format(str(i), train_text) : results_sumary[start_index + 2] / win_size})
            
        if verbose: print("Siamese model no. {0}:\tMax = {1}\tMin = {2}\tMean = {3:.5f}\tMedian = {4}\n".format(
            i, results_sumary[start_index], results_sumary[start_index + 1],
            results_sumary[start_index + 2], results_sumary[start_index + 3]))
    if verbose:
        print("------------------------------------------------------")

# -----------------------------------------------------------------------------------
