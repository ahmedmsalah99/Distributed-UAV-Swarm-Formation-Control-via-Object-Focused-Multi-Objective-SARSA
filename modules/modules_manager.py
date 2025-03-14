import numpy as np
import math
from common.utils import Point
class ObservationModule:
    def __init__(self, weight,discount):
        self.weight = weight
        self.discount = discount 

    
    def get_observations(self, env, agent_idx):
        raise NotImplementedError
    
    def calc_reward(self, env,agent_idx):
        raise NotImplementedError


class CohesionModule(ObservationModule):
    def get_observations(self, env, agent_idx):
        centroid_array = np.mean([p.to_array() for p in env.agents_positions], axis=0)
        centroid = Point(centroid_array[0],centroid_array[1])
        agent_pos = env.agents_positions[agent_idx]
        rel_pos = centroid - agent_pos
        
        distance = np.min((np.linalg.norm(rel_pos.to_array()),env.swarm_radius-1))
        bearing = (np.min((np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi,180-1))+180)/5
        return {0: [int(distance), int(bearing)]}
    
    def calc_reward(self, env,agent_idx):
        rewards = {}
        # calc distance
        centroid_array = np.mean([p.to_array() for p in env.agents_positions], axis=0)
        centroid = Point(centroid_array[0],centroid_array[1])
        agent_pos = env.agents_positions[agent_idx]
        rel_pos = centroid - agent_pos
        distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist))*10

        rewards[0] = 1 if distance < env.swarm_radius else -1
        return rewards

class AlignmentModule(ObservationModule):
    def get_observations(self, env, agent_idx):
        avg_heading = np.mean(env.agents_angles)
        # agents angle range (-30 - -150), avg_heading has the same range
        # difference range is from 0 to 150
        heading_diff = (int(env.agents_angles[agent_idx] - avg_heading)+360-1)//10
        return {0: heading_diff}
    
    def calc_reward(self, env, agent_idx):
        # 2 * (0.5-(1/pi)*heading_diff) 
        # to get 1/pi we multiply by pi/180 and then divide by pi because the difference is in degrees
        avg_heading = np.mean(env.agents_angles)
        heading_diff = round(env.agents_angles[agent_idx] - avg_heading)
        return {0: 2 * (0.5 - abs(heading_diff) / 180)} 

class CollisionModule(ObservationModule):
    def __init__(self, weight,discount,away_thresh,close_thresh):
        super().__init__(weight,discount)
        self.away_thresh = away_thresh
        self.close_thresh = close_thresh
    
    def get_refined_dist(self,dist,max_dist):
        k1,k2 = 10,5
        quantized_distance = int(k1 * np.log(1 + k2 * dist))
        max_quantized_dist = int(k1 * np.log(1 + k2 * max_dist))
        quantized_dist = (quantized_distance/max_quantized_dist)*max_dist
        return quantized_dist
    
    def get_observations(self, env, agent_idx):
        agent_pos = env.agents_positions[agent_idx]
        agent_heading = env.agents_angles[agent_idx]
        obs = {}
        
        obstacle_idx = 0
        for i, pos in enumerate(env.agents_positions):
            if i == agent_idx:
                continue
            rel_pos = pos - agent_pos
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist-1))*10
            bearing = (np.min((np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi,180-1))+180)/5
            heading_diff = (int(agent_heading - env.agents_angles[i]) + 360-1) // 10

            quantized_distance = self.get_refined_dist(distance,env.max_dist)
            obs[obstacle_idx] = [int(quantized_distance), int(bearing), heading_diff]
            obstacle_idx += 1
        return obs
    
    def calc_reward(self, env, agent_idx):
        rewards = {}
        obstacle_idx = 0
        agent_pos = env.agents_positions[agent_idx]
        for i,pos in enumerate(env.agents_positions):
            if i == agent_idx:
                continue
            rel_pos = pos - agent_pos
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist))
            if distance < self.close_thresh:
                rewards[obstacle_idx] = -100
            elif distance < self.away_thresh:
                rewards[obstacle_idx] = -1
            else:
                rewards[obstacle_idx] = 0
            obstacle_idx+=1
        return rewards

class TargetSeekModule(ObservationModule):
    def __init__(self, weight,discount,thresh):
        super().__init__(weight,discount)
        self.thresh = thresh
    
    def get_observations(self, env, agent_idx):
        agent_pos = env.agents_positions[agent_idx]
        target_pos = env.waypoints_positions[env.current_waypoint_idx]
        rel_pos = target_pos - agent_pos
        distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist-1))*10
        bearing = (np.min((np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi,180-1))+180)
        angle = (env.agents_angles[agent_idx] + 180)
        diff = (angle - bearing + 360)/10
        return {0: [int(distance), (int(diff))]}
    
    def calc_reward(self, env,agent_idx):
        rewards = {}
        way_point_dist = env.agents_positions[agent_idx].dist(env.waypoints_positions[env.current_waypoint_idx])
        rewards[0] = 10 if way_point_dist < self.thresh else -1
        return rewards

class ObstacleAvoidanceModule(ObservationModule):
    def __init__(self, weight,discount,away_thresh,close_thresh):
        super().__init__(weight,discount)
        self.away_thresh = away_thresh
        self.close_thresh = close_thresh
        
    def get_observations(self, env, agent_idx):
        agent_pos = env.agents_positions[agent_idx]
        obs = {}
        for i, pos in enumerate(env.obstacles_positions):
            rel_pos = pos - agent_pos
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist-1))*10
            bearing = (np.min((np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi,180-1))+180)/5
            obs[i] = [int(distance), int(bearing)]
        return obs
    
    def calc_reward(self, env, agent_idx):
        rewards = {}
        agent_pos = env.agents_positions[agent_idx]
        for i, pos in enumerate(env.obstacles_positions):
            rel_pos = pos - agent_pos
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist-1))
            if distance < self.close_thresh:
                rewards[i] = -100
            elif distance < self.away_thresh:
                rewards[i] = -1
            else:
                rewards[i] = 0
        return rewards
