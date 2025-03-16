from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete
import os
class SARSALearning:
    # INPUTS: 
    # env - Environment object
    # alpha - step size 
    # gamma - discount rate
    # T - The higher the more random actions will happen
    # numberEpisodes - total number of simulation episodes
     
    def __init__(self,env,modules,training_params,training_mode=True,Qmatrix_path_dict=None):
                #  ,alpha,T,number_episodes,Qmatrix_path_dict=None,training_mode=True,train_plot_title="Phase 1",save_folder = ".",window_size=20):
        # Initializations
        self.env=env
        self.number_episodes= training_params.get("number_episodes",1000)
        self.alpha=training_params.get("alpha",0.1)
        self.T = training_params.get("T",1) 
        self.curr_T = self.T
        self.modules = modules
        self.train_plot_title = training_params.get("train_plot_title","")
        self.training_mode = training_mode
        self.save_folder = training_params.get("save_folder",".")
        self.window_size = training_params.get("window_size",20)
        # Initialize 
        self.state_space=env.observation_space()
        self.action_n=env.action_space().n
        
        
        self.Qmatrix = {}
        for mod in self.state_space:
            # Qmatrix_path_dict = modules[mod]
            if Qmatrix_path_dict is not None and mod in Qmatrix_path_dict:
                loaded_Qmatrix = np.load(Qmatrix_path_dict[mod],allow_pickle=True).item()
                print(loaded_Qmatrix)
                self.Qmatrix[mod] = loaded_Qmatrix[mod]
            else:
                current_state_space = self.state_space[mod]
                if isinstance(current_state_space,MultiDiscrete):
                    whole_shape = list(current_state_space.nvec) + [self.action_n]
                else:
                    whole_shape = [current_state_space.n] + [self.action_n]
                self.Qmatrix[mod] = np.zeros(whole_shape)
        
         
    # this function selects an action on the basis of the current state 
    # INPUTS: 
    # state - state for which to compute the action
    # index - index of the current episode
    def select_action(self,state,action_mask):
        
        Q_GM = self.calc_Q_GM(state)
        Q_GM = Q_GM[action_mask]

        
        exp_term = np.exp(Q_GM/self.curr_T)
        
        dom  = np.sum(exp_term)

        if dom == 0:
            P_sa = [1/len(Q_GM) for _ in Q_GM]
        elif dom == float("inf"):
            P_sa = [0 for _ in Q_GM]
            P_sa[np.argmax(Q_GM)] = 1
        else:
            P_sa = exp_term / dom
        actions = np.array(range(self.action_n))[action_mask]
        # we return the index after sampling according to probailities 
        if len(actions) == 0:
            # raise Exception("No valid actions can be taken")
            return None
        return np.random.choice(actions,p=P_sa)
    
    def calc_Q_GM(self,state_s):
        Q_GM = np.zeros(self.action_n)
        # calc Q_GMs
        for mod in self.modules:
            Q_m = np.zeros(self.action_n)
            for obj in state_s[mod]:
                curr_state_s = state_s[mod][obj]
                if  isinstance(curr_state_s,list):
                    curr_state_s = tuple(curr_state_s)
                Q_m += self.Qmatrix[mod][curr_state_s]
                if np.isnan(Q_m).any():
                    print("fault detected")

            Q_GM += self.modules[mod].weight * Q_m
        return Q_GM
    
    def update_Qmatrix(self,state_s,state_sprime,action_a,action_aprime,rewards,termination):
        rewards_acc = 0
        for mod in self.modules:
            gamma = self.modules[mod].discount
            for obj in state_s[mod]:
                if rewards[mod][obj] is None:
                    continue

                # get state_s, state_sprime, reward for the module and object 
                curr_state_sprime = state_sprime[mod][obj]
                curr_state_s = state_s[mod][obj]
                reward = rewards[mod][obj]
                rewards_acc += reward
                
                # if rewards_acc > 50:
                #     print(rewards_acc)
                if isinstance(curr_state_sprime,list):
                        curr_state_sprime = tuple(curr_state_sprime)
                if isinstance(curr_state_s,list):
                    curr_state_s = tuple(curr_state_s)
                if not termination:
                    # SARSA formula
                    
                    error= reward+gamma*self.Qmatrix[mod][curr_state_sprime][action_aprime]-gamma*self.Qmatrix[mod][curr_state_s][action_a]
                    self.Qmatrix[mod][curr_state_s][action_a]=self.Qmatrix[mod][curr_state_s][action_a]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error = reward - self.Qmatrix[mod][curr_state_s][action_a]
                    self.Qmatrix[mod][curr_state_s][action_a]=self.Qmatrix[mod][curr_state_s][action_a]+self.alpha*error
        return rewards_acc
    
    def simulate_episodes(self):
        # Create the plot
        rewards_array = []
        episode_indices = []
        average_rewards = []
        if self.training_mode:
            # Plotting Initialization
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            line, = ax.plot(episode_indices, average_rewards, color='navy', linewidth=2, label="Reward")        

            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Reward')
            ax.set_title(self.train_plot_title)
            ax.grid(alpha=0.3)
        
        # here we loop through the episodes
        for index_episode in range(self.number_episodes):
            if self.training_mode:
                self.curr_T = self.T + (1-self.T)*(index_episode/self.number_episodes)
            else:
                self.curr_T = 1
            # reset the environment at the beginning of every episode
            observations,info = self.env.reset()
            terminated = [False for _ in self.env.agents]
            rewards_acc = 0
            actions = []
            states = []
            print("Simulating episode {}".format(index_episode))

            # initialize states, and actions
            for i in self.env.agents:
                # print(observations)
                state_s = observations[i]
                states.append(state_s)
                # select an action on the basis of the initial state
                action_mask = info[i]['action_mask']
                action_a = self.select_action(state_s,action_mask)
                # if action_a is None:
                #     raise Exception("No actions can be taken at the starting position!")
                actions.append(action_a)
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            while not np.all(terminated):
                for i in self.env.agents:
                    # agent already terminated
                    if terminated[i]:
                        continue
                    # move the agent
                    if actions[i] is None:
                        continue
                    observations, rewards, termination, truncation, info = self.env.step(actions[i],i)
                    # if np.any([rewards['OBS'][key] < -1  for key in rewards['OBS'] if rewards['OBS'][key] is not None ]) or np.any([rewards['COL'][key] < -1 for key in rewards['COL'] if rewards['COL'][key] is not None]):
                    #     print("collision")
                    action_mask = info['action_mask']
                    # time up
                    if  truncation:
                        terminated[i] = True
                        continue
                    # current action and state
                    action_a = actions[i]
                    state_s = states[i]

                    # next state and action
                    state_sprime = observations
                    action_aprime = self.select_action(state_sprime,action_mask)
                    new_action_taken = True
                    if action_aprime is None:
                        action_aprime = action_a
                        new_action_taken = False
                    # if rewards["OBS"]:
                    # Update only if the mask is all valid
                    if  self.training_mode:# and np.all(action_mask)
                        rewards_acc += self.update_Qmatrix(state_s,state_sprime,action_a,action_aprime,rewards,termination)
                    # states_visited.add(tuple(states[i]["TGT"][0]))
                    # update current actions, states, and termination state
                    actions[i] = action_aprime
                    states[i] = state_sprime
                    terminated[i] = termination
                    if not self.training_mode:
                        self.env.render()
                        sleep(0.01/self.env.agents_num)
                    # self.env.render()
                    # sleep(0.1/self.env.agents_num)
            rewards_array.append(rewards_acc/self.env.agents_num)
            # print("rewards",rewards_acc)
            # print("time",self.env.timestep)
            # print("states_visited",len(states_visited))
            
            
            # Calculate the average reward for the last 200 episodes
            if (index_episode+1) % self.window_size == 0 and self.training_mode:
                rolling_mean = np.convolve(rewards_array, np.ones(self.window_size)/self.window_size, mode='valid')
                # rolling_std = np.std(rewards_array[:len(rolling_mean)])
                x_values = np.arange(len(rolling_mean))
                # Update the plot
                line.set_xdata(x_values)
                line.set_ydata(rolling_mean)
                ax.relim()  # Recalculate limits
                ax.autoscale_view(True, True, True)  # Rescale the view
                plt.draw()
                plt.pause(0.2)  # Pause to allow the plot to update
                save_path_graph = os.path.join(self.save_folder,"training_"+str(index_episode)+".png")
                save_path_matrix = os.path.join(self.save_folder,"Qmatrix_"+str(index_episode)+".npy")
                np.save(save_path_matrix, self.Qmatrix)
                fig.savefig(save_path_graph, dpi=300)
        if self.training_mode:
            # Keep the plot open after the loop finishes
            plt.ioff()
            plt.show()
