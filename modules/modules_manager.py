import numpy as np
import math
from common.utils import Point
class ObservationModule:
    def __init__(self, weight,discount):
        self.weight = weight
        self.discount = discount 

    
    def get_observations(self, env, agent_idx):
        raise NotImplementedError
    
    def calc_reward(self, observations):
        raise NotImplementedError


class CohesionModule(ObservationModule):
    def get_observations(self, env, agent_idx):
        centroid = Point(np.mean([p.to_array() for p in env.agents_positions], axis=0))
        agent_pos = env.agents_positions[agent_idx]
        rel_pos = centroid - agent_pos
        distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist))
        bearing = np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi
        return {0: [int(distance * 4 - 1), round(bearing + 360)]}
    
    def calc_reward(self, observations):
        rewards = {}
        swarm_radius = np.sqrt(len(observations)) * 5  # Example scale factor
        for obj, (dist, _) in observations.items():
            rewards[obj] = 1 if dist < swarm_radius else -1
        return rewards

class AlignmentModule(ObservationModule):
    def get_observations(self, env, agent_idx):
        avg_heading = np.mean(env.agents_angles)
        heading_diff = round(env.agents_angles[agent_idx] - avg_heading + 360)
        return {0: heading_diff}
    
    def calc_reward(self, observations):
        # 2 * (0.5-(1/pi)*heading_diff) 
        # to get 1/pi we multiply by pi/180 and then divide by pi because the difference is in degrees
        return {obj: 2 * (0.5 - abs(obs) / 180) for obj, obs in observations.items()} 

class CollisionModule(ObservationModule):
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
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist))
            bearing = np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi
            heading_diff = round(agent_heading - env.agents_angles[i] + 360)

            quantized_distance = self.get_refined_dist(distance,env.max_dist)
            obs[obstacle_idx] = [int(quantized_distance*4 - 1), round(bearing + 360), heading_diff]
            obstacle_idx += 1
        return obs
    
    def calc_reward(self, observations):
        rewards = {}
        for obj, (dist,_,_) in observations.items():
            if dist < 0.2:
                rewards[obj] = -100
            elif dist < 1.5:
                rewards[obj] = -1
            else:
                rewards[obj] = 0
        return rewards

class TargetSeekModule(ObservationModule):
    def __init__(self, weight,discount,thresh):
        super().__init__(weight,discount)
        self.thresh = thresh
    
    def get_refined_dist(self,dist,max_dist):
        k1,k2 = 10,0.1
        quantized_distance = int(k1 * np.log(1 + k2 * dist))
        max_quantized_dist = int(k1 * np.log(1 + k2 * max_dist))
        quantized_dist = (quantized_distance/max_quantized_dist)*max_dist
        return quantized_dist
    # @profile
    def get_observations(self, env, agent_idx):
        agent_pos = env.agents_positions[agent_idx]
        target_pos = env.waypoints_positions[env.current_waypoint_idx]
        rel_pos = target_pos - agent_pos

        distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist-1))
        # distance = self.get_refined_dist(distance,env.max_dist)
        bearing = np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi 
        return {0: [int(np.floor(distance)), (int(np.floor(bearing + 180))//8)-1]}
    
    def calc_reward(self, way_point_dist):
        rewards = {}
        rewards[0] = 10 if way_point_dist < self.thresh else -1
        return rewards

class ObstacleAvoidanceModule(ObservationModule):
    def get_observations(self, env, agent_idx):
        agent_pos = env.agents_positions[agent_idx]
        obs = {}
        for i, pos in enumerate(env.obstacles_positions):
            rel_pos = pos - agent_pos
            distance = np.min((np.linalg.norm(rel_pos.to_array()),env.max_dist))
            bearing = np.arctan2(rel_pos.y, rel_pos.x) * 180 / np.pi
            obs[i] = [int(distance * 4 - 1), round(bearing + 360)]
        return {0: obs}
    
    def calc_reward(self, observations):
        rewards = {}
        for obj, (dist,_) in observations.items():
            if dist < 1:
                rewards[obj] = -100
            elif dist < 3:
                rewards[obj] = -1
            else:
                rewards[obj] = 0
        return rewards
