import numpy as np
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

agents_positions = [
    Point(11.42387517183825, 19.906485633166927),
    Point(9.064214417688811, 21.331673949649595),
    Point(10.412206063624428, 21.940674256428665),  # agent_idx = 2
    Point(8.655895395092056, 22.50098569797893),
    Point(10.38887492439058, 21.99409809302786),
    Point(11.737751878218726, 21.39371280976829),
    Point(8.416291968824893, 21.8399487089115),
    Point(9.083772781512161, 22.277743495003023),
    Point(9.869409200852308, 20.251889246956298),
    Point(11.185154512933074, 20.70723729306866)
]

def check_object_collision(agent_idx, agent_new_pos, agents_positions, collision_dist=0.355):
    truth_array = [pos.dist(agent_new_pos) <= collision_dist for pos in agents_positions]
    dists = [ pos.dist(agent_new_pos) for pos in agents_positions]
    truth_array[agent_idx] = True  # Exclude own position
    print(dists)
    print(truth_array)
    return np.sum(truth_array) != 1

# Run the function
agent_idx = 2
agent_new_pos = agents_positions[2]
result = check_object_collision(agent_idx, agent_new_pos, agents_positions, collision_dist=0.355)
print("Collision detected:", result)
