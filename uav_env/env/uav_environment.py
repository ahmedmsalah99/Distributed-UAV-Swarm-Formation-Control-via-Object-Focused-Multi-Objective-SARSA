import functools
import random
from copy import copy
from common.utils import Point
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
import pygame
from pettingzoo import ParallelEnv


class UAVEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name": "uav_environment_v0",
    }

    def __init__(self,agents_num,modules,env_params,render_mode = "gui"):
        """The init method takes in environment arguments.
        """
        # Meta informations about the environment
        self.timeout = env_params.get("timeout",10000)
        self.waypoint_in_radius = env_params.get("waypoint_in_radius",None)
        self.update_waypoint= env_params.get("update_waypoint",True)
        self.max_dist = env_params.get("max_dist",1.0)
        self.modules = modules
        self.is_urban = False
        self.render_mode = render_mode
        self.screen = None 
        self.agents_num = agents_num
        self.row_size = 5
        self.size = Point(env_params.get("x_size",100),env_params.get("y_size",100))
        self.waypoints_num = env_params.get("waypoints_num",1)
        self.obstacles_num = env_params.get("obstacles_num",0)
        self.waypoint_dist_thesh = env_params.get("waypoint_dist_thesh",1.5)
        self.cell_size = 10
        self.actions = [-60,-45,-30,-15,-8,0,8,15,30,45,60]
        self.swarm_radius = np.sqrt(agents_num) * 0.25
        self.resolution = 1.0
        self.agent_speed = 0.2
        self.collision_dist = self.agent_speed/2
        half_point = Point(self.size.x//2,self.size.y)
        self.spawn_center = half_point - Point(env_params.get("spawn_center_x",1.0),env_params.get("spawn_center_y",1.0)) 
        self.border_x_down = self.spawn_center.x + self.swarm_radius
        residual = np.max((self.border_x_down - (self.size.x-1),0))
        self.border_x_down = np.max((self.border_x_down - residual,0))
        self.border_x_up = np.max((self.spawn_center.x - self.swarm_radius - residual,self.size.x-1))

        self.border_y_right = self.spawn_center.y+self.swarm_radius
        residual = np.max((self.border_y_right - (self.size.y-1) ,0))
        self.border_y_right = np.max((self.border_y_right - residual,0))
        self.border_y_left = np.max((self.spawn_center.y - self.swarm_radius - residual,self.size.y-1))
        # other objects positions


        # initializations
        self.timestep = None

        # agents initialization
        self.possible_agents = np.array(list(range(agents_num)))
        self.agents_positions = np.array([Point(0,0) for _ in self.possible_agents])
        self.agents_angles = np.array([0 for _ in self.possible_agents])
        self.grid = np.zeros((self.size.x,self.size.y))
        
        # obstacles initialization
        self.waypoints_positions = np.array([])
        self.obstacles_positions = np.array([])

    def generate_obstacles(self):
        obstacles = np.array([])
        for _ in range(self.obstacles_num):
            x = random.uniform(0, self.size.x - 1)
            y = random.uniform(0,self.border_y_left)
            candidate = Point(x,y)
            while candidate in self.waypoints_positions or candidate in self.agents_positions:
                y = random.uniform(0,self.border_y_left)
                candidate = Point(x,y)
            obstacles = np.append(obstacles,candidate)
        return obstacles
    
    def generate_waypoints(self):
        waypoints = np.array([])
        half_grid = self.size.x // 2
        # step = self.size.y //(self.waypoints_num+1)
        step = (self.border_y_left) // (self.waypoints_num+1)
        # start_x = np.max([point.x for point in self.agents_positions])
        if self.waypoint_in_radius is not None:
            centroid_array = np.mean([p.to_array() for p in self.agents_positions], axis=0)
            centroid = Point(centroid_array[0],centroid_array[1])
            for i in range(self.waypoints_num):
                x = random.uniform(centroid.x-self.waypoint_in_radius, centroid.x+self.waypoint_in_radius)
                y = random.uniform(centroid.y-self.waypoint_in_radius, centroid.y+self.waypoint_in_radius)
                x = min(max(x,0),self.size.x-1)
                y = min(max(y,0),self.size.y-1)
                candidate = Point(x,y)
                waypoints = np.append(waypoints,candidate)
            return waypoints


        for i in range(self.waypoints_num):
            if self.waypoints_num == 1:
                x = random.uniform(0, self.size.x - 1)
            else:
                # Alternate between right and left halves of the grid
                if i % 2 == 0:
                    # Right half
                    x = random.uniform(half_grid, self.size.x - 1)
                else:
                    # Left half
                    x = random.uniform(0, half_grid - 1)
            # in vertical order
            # y = random.randint(max(step,(step)*(i+2)), step*(i+1)-1)
            y = random.uniform(step*(i), step*(i+1)-1)
            candidate = Point(x,y)
            # not in obstacle position
            while candidate in self.obstacles_positions:
                y = random.uniform(step*(i), step*(i+1)-1)
                # y = random.randint(max(step,step*(i+2)), step*(i+1)-1)
                candidate = Point(x, y)
            
            waypoints = np.append(waypoints,candidate)
        return waypoints
    
    def generate_agent_pos(self):
        x = random.uniform(self.border_x_up,self.border_x_down)
        y = random.uniform(self.border_y_left,self.border_y_right)

        return Point(x,y)
    
    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.timestep = 0
        self.current_waypoint_idx = 0
        self.agents = list(copy(self.possible_agents))


        # self.agents_positions = np.array([self.start + Point(agent%self.row_size+random_x,agent//self.row_size) for agent in self.possible_agents])
        self.agents_positions = np.array([])
        for agent in self.possible_agents:
            candidate = self.generate_agent_pos()
            if agent == 0:
                self.agents_positions = np.append(self.agents_positions,candidate)
                continue
            dists = [self.agents_positions[i].dist(candidate) for i in range(0,agent)]
            min_dist = np.min(dists)
            while min_dist < self.resolution * 2:
                candidate = self.generate_agent_pos()
                dists = [self.agents_positions[i].dist(candidate) for i in range(0,agent)]
                min_dist = np.min(dists)
            
            self.agents_positions = np.append(self.agents_positions,candidate)
        # self.agents_angles = np.array([np.random.randint(0,360) for _ in self.possible_agents])
        self.agents_angles = np.array([0 for _ in self.possible_agents])
        
        # obstacles and waypoints initialization
        self.waypoints_positions = self.generate_waypoints()
        self.obstacles_positions = self.generate_obstacles()
        # Get observations
        observations = {}
        for agent_idx in self.agents:
            observations[agent_idx] = {}
            for mod in self.modules:
                observations[agent_idx][mod] = self.modules[mod].get_observations(self,agent_idx)
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {agent:{"action_mask":self.get_actions_masks(agent)} for agent in self.agents}
        return observations, infos
    
    def get_actions_masks(self,agent_idx):
        
        masks = np.array([
             self.validate_pos(agent_idx,self.update_state(agent_idx,i)[0])[0]
              for i,_ in enumerate(self.actions)])

        return masks
    # def get_observations(self,agent_idx):
    #     observations = {
    #         i:{"TGT":(None,None)}
    #         for i,agent in 
    #         enumerate(self.agents)}
        
    #     curr_waypoint = self.waypoints_positions[self.current_waypoint_idx]
    #     for i in self.agents:
    #         waypoint_angle_diff = self.agents_angles[i]-self.agents_positions[i].angle(curr_waypoint) + 360
    #         waypoint_dist = np.round(self.agents_positions[i].dist(curr_waypoint)*4)
    #         observations[i]['TGT'] = (waypoint_dist,waypoint_angle_diff)
    #     return observations
    # @profile
    def update_state(self,agent_idx,action):
        old_pos = self.agents_positions[agent_idx]
        new_angle = (self.agents_angles[agent_idx] + self.actions[action])
        # new_angle = (self.actions[action] - 90)*(np.pi/180)
        # angle_sign = np.sign(new_angle)
        if new_angle >= 150:
            new_angle = 150
        elif new_angle < -150:
            new_angle = -150
        new_angle_radians = (new_angle-90)*(np.pi/180)
        movement = Point(self.agent_speed*np.cos(new_angle_radians),self.agent_speed*np.sin(new_angle_radians))
        new_pos = old_pos + movement
        return new_pos,new_angle
    # Checks
    # @profile
    def check_boundries(self,agent_new_pos:Point):
        if agent_new_pos.x < 0 or agent_new_pos.y <0:
            return False
        
        if  agent_new_pos.x > self.size.x -1 or agent_new_pos.y > self.size.y -1:
            return False
        return True

    # @profile
    def check_object_collision(self,agent_idx,agent_new_pos:Point):
        if self.agents_num == 1:
            return True
        truth_array  = [pos.dist(agent_new_pos) <= self.collision_dist for pos in self.agents_positions]
        # exclude object's position itself
        truth_array[agent_idx] = True
        return np.sum(truth_array) == 1
    
    # @profile
    def check_obstacle_collision(self,agent_new_pos):
        truth_array = [False if pos.dist(agent_new_pos) <= self.collision_dist else True for pos in self.obstacles_positions]
        return np.all(truth_array)
    
    def validate_pos(self,agent_idx,agent_new_pos):
        no_hit_boundry = self.check_boundries(agent_new_pos)
        no_hit_object = self.check_object_collision(agent_idx,agent_new_pos)
        no_hit_obstacle = self.check_obstacle_collision(agent_new_pos)

        return no_hit_boundry and no_hit_object and no_hit_obstacle,{'hit_boundry':not no_hit_boundry,'hit_object':not no_hit_object,'hit_obstacle':not no_hit_obstacle}

    def calc_reward(self,agent_idx):
        rewards = {}
        for mod in self.modules:
            rewards[mod] = self.modules[mod].calc_reward(self,agent_idx)
        return rewards
    def resolve_boundry_situation(self,agent_idx):
        pos = self.agents_positions[agent_idx]
        x = max(min(pos.x,self.size.x-self.agent_speed),self.agent_speed)
        y = max(min(pos.y,self.size.y-self.agent_speed),self.agent_speed)
        return Point(x,y)
    def step(self, action,agent_idx):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        
                
        # Execute actions
        # update angle
        termination = False
        # update position
        new_pos,new_angle = self.update_state(agent_idx,action)
        pos_valid, hits = self.validate_pos(agent_idx,new_pos)
        masks = self.get_actions_masks(agent_idx)
        if pos_valid:
            self.agents_positions[agent_idx] = new_pos
            self.agents_angles[agent_idx] = new_angle
        if np.all(masks==False) and hits["hit_boundry"]:
            new_pos = self.resolve_boundry_situation(agent_idx)
            self.agents_positions[agent_idx] = new_pos
            self.agents_angles[agent_idx] = new_angle
            masks = self.get_actions_masks(agent_idx)
        if np.all(masks==False):
            termination = True
            # x = 0
            # y = 0
            # if self.agents_positions[agent_idx].x >= self.size.x - self.collision_dist:
            #     x = self.size.x - self.collision_dist
            # elif self.agents_positions[agent_idx].x < self.collision_dist:
            #     x = self.collision_dist
            # if self.agents_positions[agent_idx].y >= self.size.y - self.collision_dist:
            #     y = self.size.y - self.collision_dist
            # elif self.agents_positions[agent_idx].y < self.collision_dist:
            #     y = self.collision_dist

            # self.agents_positions[agent_idx] += Point(x,y)
            # self.agents_angles[agent_idx] = new_angle


        rewards = self.calc_reward(agent_idx)

        # Update way point and check termination conditions
        if self.waypoints_num > 0:
            way_point_dist = self.agents_positions[agent_idx].dist(self.waypoints_positions[self.current_waypoint_idx])
            in_range = way_point_dist < self.waypoint_dist_thesh
            
            if in_range and self.update_waypoint:
                if self.current_waypoint_idx == self.waypoints_num - 1:
                    termination |= in_range
                else:
                    self.current_waypoint_idx += 1


        
        observations = {}
        for mod in self.modules:
            observations[mod] = self.modules[mod].get_observations(self,agent_idx)
        # Check truncation conditions (overwrites termination conditions)
        truncation =  False
        if self.timestep > self.timeout *self.agents_num:
            truncation = True
        self.timestep += 1


        
        
        infos = {"action_mask":masks}
        return observations, rewards, termination, truncation, infos

    def render(self):
        """Renders the environment."""
        resolution_scale = int(1/self.resolution)
        x_size = self.size.x*resolution_scale
        y_size = self.size.y*resolution_scale
        self.grid = np.full((y_size,x_size)," ")
        for obstacle in self.obstacles_positions:
            obstacle = obstacle * resolution_scale
            self.grid[int(obstacle.y)][int(obstacle.x)] = "X"
            
        for i in range(self.current_waypoint_idx,len(self.waypoints_positions)):
            waypoint = self.waypoints_positions[i]*resolution_scale
            self.grid[int(waypoint.y)][int(waypoint.x)] = "G"

        for agent_pos in self.agents_positions:
            y = np.min((agent_pos.y*resolution_scale,y_size-1))
            x = np.min((agent_pos.x*resolution_scale,x_size-1))
            if self.grid[int(y)][int(x)] == "O" or self.grid[int(y)][int(x)] == "X":
                print("2 over each other") 
            self.grid[int(y)][int(x)] = "O"

        if self.render_mode == "gui":
            """Renders the environment using pygame."""

            if self.screen is None:
                # Initialize pygame and the screen if not already done
                pygame.init()
                self.screen = pygame.display.set_mode((x_size * self.cell_size, y_size * self.cell_size))

            # Handle pygame events (e.g., window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.screen = None
                    return

            # Clear the screen
            self.screen.fill((230, 230, 230))  # White background
            
            # Draw the grid
            for y in range(y_size):
                for x in range(x_size):
                    # Draw grid lines
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray grid lines

                    cell = self.grid[y][x]
                    if cell == "X":
                         pygame.draw.rect(self.screen, (100, 100, 100), rect)  # Dark gray for obstac
                    elif cell == "G":
                        center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                        pygame.draw.circle(self.screen, (255, 0, 0), center, self.cell_size // 3) 
                    elif cell == "O":
                        # Draw a triangle (polygon) for the agent
                        points = [
                            (x * self.cell_size + self.cell_size // 2, y * self.cell_size),  # Top point
                            (x * self.cell_size, y * self.cell_size + self.cell_size),  # Bottom-left point
                            (x * self.cell_size + self.cell_size, y * self.cell_size + self.cell_size)  # Bottom-right point
                        ]
                        pygame.draw.polygon(self.screen, (0, 128, 255), points)  # Blue triangle for agents


            # Update the display
            pygame.display.flip()
        else:
            print(f"{self.grid} \n")

   

         
    def clear(self):
        pygame.quit()
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        temp_dict = {}
        if "TGT" in self.modules:
            # temp_dict["TGT"] = MultiDiscrete([self.max_dist,(360)*(2/5)])
            temp_dict["TGT"] = MultiDiscrete([self.max_dist*10,360*(2/10)])
        if "COH" in self.modules:
            temp_dict["COH"] = MultiDiscrete([self.swarm_radius*10,360/5])
        if "OBS" in self.modules:
            temp_dict["OBS"] = MultiDiscrete([self.max_dist*10,360/5])
        if "COL" in self.modules:
            temp_dict["COL"] = MultiDiscrete([self.max_dist*10,360/5,360/5])
        if "ALN" in self.modules:
            temp_dict["ALN"] = MultiDiscrete([360/5])
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(temp_dict)
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return Discrete(11)