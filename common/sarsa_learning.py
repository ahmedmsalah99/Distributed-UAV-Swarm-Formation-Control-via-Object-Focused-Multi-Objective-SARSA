import numpy as np
import matplotlib.pyplot as plt

class SARSALearning:
    # INPUTS: 
    # env - Environment object
    # alpha - step size 
    # gamma - discount rate
    # T - The higher the more random actions will happen
    # numberEpisodes - total number of simulation episodes
     
    def __init__(self,env,modules,alpha,T,number_episodes):
        # Initializations
        self.env=env
        self.number_episodes=number_episodes
        self.alpha=alpha
        self.T = T 
        self.curr_T = T
        self.modules = modules
        

        # Initialize 
        self.state_space=env.observation_space()
        self.action_n=env.action_space().n
        
        self.Qmatrix = {}
        for mod in self.state_space:
            current_state_space = self.state_space[mod]
            whole_shape = list(current_state_space.nvec) + [self.action_n]
            self.Qmatrix[mod] = np.zeros(whole_shape)
        
         
    # this function selects an action on the basis of the current state 
    # INPUTS: 
    # state - state for which to compute the action
    # index - index of the current episode
    def select_action(self,state,index,action_mask):
        # first 100 episodes we select completely random actions to avoid being stuck
        if index<0:
            return np.random.choice(self.action_n)   
        # otherwise, we are selecting greedy actions
        else:
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
            return np.random.choice(actions,p=P_sa)
    
    def calc_Q_GM(self,state_s):
        Q_GM = np.zeros(self.action_n)
        # calc Q_GMs
        for mod in self.modules:
            Q_m = np.zeros(self.action_n)
            for obj in state_s[mod]:
                curr_state_s = state_s[mod][obj]
                Q_m += self.Qmatrix[mod][tuple(curr_state_s)]
                if np.isnan(Q_m).any():
                    print("fault detected")

            Q_GM += self.modules[mod].weight * Q_m
        return Q_GM
    # @profile
    def simulate_episodes(self):
        # Create the plot
        rewards_array = []
        episode_indices = []
        average_rewards = []
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot(episode_indices, average_rewards, color='navy', linewidth=2, label="Reward")        

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Phase 1')
        ax.grid(alpha=0.3)
        # ax.set_xticks(fontsize=10)
        # ax.set_yticks(fontsize=10)
        # here we loop through the episodes
        for index_episode in range(self.number_episodes):
            states_visited = set()
            self.curr_T = self.T + (1-self.T)*(index_episode/self.number_episodes)
            # reset the environment at the beginning of every episode
            observations,info = self.env.reset()
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
                action_a = self.select_action(state_s,index_episode,action_mask)
                actions.append(action_a)
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminated = [False for _ in self.env.agents]
            rewards_acc = 0
            while not np.all(terminated):
                
                for i in self.env.agents:
                    
                    # agent already terminated
                    if terminated[i]:
                        continue
                    # move the agent
                    observations, rewards, termination, truncation, info = self.env.step(actions[i],i)
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
                    action_aprime = self.select_action(state_sprime,index_episode,action_mask)

                    # Update only if the mask is all valid
                    if np.all(action_mask):
                        for mod in self.modules:
                            gamma = self.modules[mod].discount
                            for obj in observations[mod]:
                                # get state_s, state_sprime, reward for the module and object 
                                curr_state_sprime = state_sprime[mod][obj]
                                curr_state_s = state_s[mod][obj]
                                reward = rewards[mod][obj]
                                rewards_acc += reward
                                if rewards_acc > 50:
                                    print(rewards_acc)
                                if not termination:
                                    # SARSA formula
                                    error= reward+gamma*self.Qmatrix[mod][tuple(curr_state_sprime)][action_aprime]-gamma*self.Qmatrix[mod][tuple(curr_state_s)][action_a]
                                    self.Qmatrix[mod][tuple(curr_state_s)][action_a]=self.Qmatrix[mod][tuple(curr_state_s)][action_a]+self.alpha*error
                                else:
                                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                                    error = reward - self.Qmatrix[mod][tuple(curr_state_s)][action_a]
                                    self.Qmatrix[mod][tuple(curr_state_s)][action_a]=self.Qmatrix[mod][tuple(curr_state_s)][action_a]+self.alpha*error
                    states_visited.add(tuple(states[i]["TGT"][0]))
                    # update current actions, states, and termination state
                    actions[i] = action_aprime
                    states[i] = state_sprime
                    terminated[i] = termination
                    # if index_episode > 1:
                    #     self.env.render()
            rewards_array.append(rewards_acc)
            print("rewards",rewards_acc)
            print("time",self.env.timestep)
            print("states_visited",len(states_visited))
            states_visited = set()
            
            
            # Calculate the average reward for the last 200 episodes
            if (index_episode+1) % 200 == 0:
                window_size = 200
                rolling_mean = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
                # rolling_std = np.std(rewards_array[:len(rolling_mean)])
                x_values = np.arange(len(rolling_mean))



                # avg_reward = np.mean(rewards_array[-200:])
                # average_rewards.append(avg_reward)

                # ax.collections.clear()  # Remove previous fill_between
                # ax.fill_between(x_values, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                #  color='navy', alpha=0.3)
                # Update the plot
                line.set_xdata(x_values)
                line.set_ydata(rolling_mean)
                ax.relim()  # Recalculate limits
                ax.autoscale_view(True, True, True)  # Rescale the view
                plt.draw()
                plt.pause(0.05)  # Pause to allow the plot to update
            
                np.save("Qmatrix_"+str(index_episode)+".npy", self.Qmatrix)
                fig.savefig("training_phase1_"+str(index_episode)+".png", dpi=300)

            # self.env.render()

                    
                
     
# Keep the plot open after the loop finishes
plt.ioff()
plt.show()
