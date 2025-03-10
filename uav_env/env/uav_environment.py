import functools
import random
from copy import copy
from common.utils import Point
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
import pygame
from pettingzoo import ParallelEnv


class UAVEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name": "uav_environment_v0",
    }

    def __init__(self,agents_num,modules,update_waypoint=True,render_mode = "gui",max_dist = 100,obstacles_num = 30,waypoints_num = 6,size=Point(100,100),waypoint_dist_thesh=1.5,is_urban=False):
        """The init method takes in environment arguments.
        """
        # Meta informations about the environment
        self.update_waypoint= update_waypoint
        self.max_dist = max_dist
        self.modules = modules
        self.is_urban = is_urban
        self.render_mode = render_mode
        self.screen = None 
        self.agents_num = agents_num
        self.row_size = 5
        self.size = size
        self.waypoints_num = waypoints_num
        self.obstacles_num = obstacles_num
        self.waypoint_dist_thesh = waypoint_dist_thesh
        self.cell_size = 10
        self.actions = [-60,-45,-30,-15,-8,0,8,15,30,45,60]
        # other objects positions
        self.start = Point(0,0)

        # initializations
        self.timestep = None

        # agents initialization
        self.possible_agents = np.array(list(range(agents_num)))
        self.agents_positions = np.array([Point(0,0) for _ in self.possible_agents])
        self.agents_angles = np.array([0 for _ in self.possible_agents])
        self.grid = np.zeros((size.x,size.y))
        
        # obstacles initialization
        self.waypoints_positions = np.array([])
        self.obstacles_positions = np.array([])

    def generate_obstacles(self):
        
        obstacles = np.array([])

        start_x = np.max([point.x for point in self.agents_positions])
        start_y = np.max([point.y for point in self.agents_positions])

        for _ in range(self.obstacles_num):
            x = random.randint(start_x, self.size.x - 1)
            y = random.randint(start_y, self.size.y - 1)
            candidate = Point(x,y)
            while candidate in self.waypoints_positions:
                y = random.randint(start_y, self.size.y - 1)
                candidate = Point(x,y)
            obstacles = np.append(obstacles,candidate)
        return obstacles
    
    def generate_waypoints(self):

        
        waypoints = np.array([])
        half_grid = self.size.x // 2
        step = self.size.y //self.waypoints_num

        start_x = np.max([point.x for point in self.agents_positions])
        start_y = np.max([point.y for point in self.agents_positions])

        for i in range(self.waypoints_num):
            # Alternate between right and left halves of the grid
            if i % 2 == 0:
                # Right half
                x = random.randint(half_grid, self.size.x - 1)
            else:
                # Left half
                x = random.randint(start_x, half_grid - 1)
            # in vertical order
            y = random.randint(max(start_y,step*i), step*(i+1)-1)
            candidate = Point(x,y)
            # not in obstacle position
            while candidate in self.obstacles_positions:
                y = random.randint(max(start_y,step*i), step*(i+1)-1)
                candidate = Point(x, y)
            
            waypoints = np.append(waypoints,candidate)
        return waypoints
    
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
        self.agents_positions = np.array([self.start + Point(agent%self.row_size,agent//self.row_size) for agent in self.possible_agents])
        self.agents_angles = np.array([np.random.randint(0,360) for _ in self.possible_agents])
        
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
        infos = {a: {} for a in self.agents}

        return observations, infos
    
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
    def update_position(self,old_pos,angle):
        movement = Point(0,0)
        # 45 degrees for each part
        if (angle >= 337.5 or angle < 22.5):  # right
            movement = Point(1, 0)
        elif (angle >= 22.5 and angle < 67.5):  # right-top
            movement = Point(1, 1)
        elif (angle >= 67.5 and angle < 112.5):  # top
            movement = Point(0, 1)
        elif (angle >= 112.5 and angle < 157.5):  # left-top
            movement = Point(-1, 1)
        elif (angle >= 157.5 and angle < 202.5):  # left
            movement = Point(-1, 0)
        elif (angle >= 202.5 and angle < 247.5):  # left-bot
            movement = Point(-1, -1)
        elif (angle >= 247.5 and angle < 292.5):  # bot
            movement = Point(0, -1)
        elif (angle >= 292.5 and angle < 337.5):  # right-bot
            movement = Point(1, -1)
        movement.x = movement.x * 0.5
        movement.y = movement.y * 0.5
        new_pos = old_pos + movement
        return new_pos
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
        truth_array = self.agents_positions != agent_new_pos
        # exclude object's position itself
        truth_array[agent_idx] = 1
        return np.all(truth_array)
    
    # @profile
    def check_obstacle_collision(self,agent_new_pos):
        truth_array = self.obstacles_positions != agent_new_pos
        return np.all(truth_array)
    
    def validate_pos(self,agent_idx,agent_new_pos):
        no_hit_boundry = self.check_boundries(agent_new_pos)
        no_hit_object = self.check_object_collision(agent_idx,agent_new_pos)
        no_hit_obstacle = self.check_obstacle_collision(agent_new_pos)
        return no_hit_boundry and no_hit_object and no_hit_obstacle,{'hit_boundry':not no_hit_boundry,'hit_object':not no_hit_object,'hit_obstacle':not no_hit_obstacle}

    def calc_reward(self,agent_idx):
        rewards = {}
        for mod in self.modules:
            if mod == "TGT":
                way_point_dist = self.agents_positions[agent_idx].dist(self.waypoints_positions[self.current_waypoint_idx])
                rewards[mod] = self.modules[mod].calc_reward(way_point_dist)
        return rewards
    # @profile
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
        
        
        infos = {a: {} for a in self.agents}
        # Execute actions
        # update angle
        new_angle = self.agents_angles[agent_idx]+self.actions[action]
        # self.agents_angles[agent_idx] = self.agents_angles[agent_idx]+self.actions[action]
        if new_angle > 360:
            new_angle = new_angle % 360
        elif new_angle < 0:
            new_angle = new_angle + 360
        # update position
        new_pos = self.update_position(self.agents_positions[agent_idx],new_angle)
        pos_valid, hits = self.validate_pos(agent_idx,new_pos)
        
        if pos_valid:
            self.agents_positions[agent_idx] = new_pos
            self.agents_angles[agent_idx] = new_angle
        else:
            return None, None, None, None, None

        rewards = self.calc_reward(agent_idx)
        # Update way point and check termination conditions
        termination = False
        way_point_dist = self.agents_positions[agent_idx].dist(self.waypoints_positions[self.current_waypoint_idx])
        in_range = way_point_dist < self.waypoint_dist_thesh
        
        if in_range and self.update_waypoint:
            if self.current_waypoint_idx == self.waypoints_num - 1:
                termination = in_range
            else:
                self.current_waypoint_idx += 1


        
        observations = {}
        for mod in self.modules:
            observations[mod] = self.modules[mod].get_observations(self,agent_idx)
        # Check truncation conditions (overwrites termination conditions)
        truncation =  False
        if self.timestep > 100000*self.agents_num:
            truncation = True
        self.timestep += 1
        return observations, rewards, termination, truncation, infos

    def render(self):
        """Renders the environment."""
        self.grid = np.full((self.size.x,self.size.y)," ")
        for obstacle in self.obstacles_positions:
            self.grid[obstacle.y][obstacle.x] = "X"
        for i in range(self.current_waypoint_idx,len(self.waypoints_positions)):
            waypoint = self.waypoints_positions[i]
            self.grid[waypoint.y][waypoint.x] = "G"
        for agent_pos in self.agents_positions:
            self.grid[int(agent_pos.y)][int(agent_pos.x)] = "O"

        if self.render_mode == "gui":
            """Renders the environment using pygame."""
            if self.screen is None:
                # Initialize pygame and the screen if not already done
                pygame.init()
                self.screen = pygame.display.set_mode((self.size.x * self.cell_size, self.size.y * self.cell_size))

            # Handle pygame events (e.g., window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.screen = None
                    return

            # Clear the screen
            self.screen.fill((230, 230, 230))  # White background
            
            # Draw the grid
            for y in range(self.size.y):
                for x in range(self.size.x):
                    # Draw grid lines
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray grid lines

                    cell = self.grid[y][x]
                    if cell == "X":
                         pygame.draw.rect(self.screen, (100, 100, 100), rect)  # Dark gray for obstac
                    elif cell == "G":
                        center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                        pygame.draw.circle(self.screen, (255, 223, 0), center, self.cell_size // 3) 
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
            temp_dict["TGT"] = MultiDiscrete([self.max_dist,360/8])
        if "COH" in self.modules:
            temp_dict["COH"] = MultiDiscrete([self.max_dist*4,360*2-1])
        if "OBS" in self.modules:
            temp_dict["OBS"] = MultiDiscrete([self.max_dist*4,360*2-1])
        if "COL" in self.modules:
            temp_dict["COL"] = MultiDiscrete([self.max_dist*4,360*2-1,360*2-1])
        if "ALN" in self.modules:
            temp_dict["ALN"] = Discrete([360*2-1])
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Dict(temp_dict)
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return Discrete(11)