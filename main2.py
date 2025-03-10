from uav_env.env.uav_environment import UAVEnvironment
from time import sleep
from common.utils import Point
from common.parameters_manager import ParametersManager

parameters_manager = ParametersManager('configs/train_phase1.yaml')
env = UAVEnvironment(agents_num=2,obstacles_num=5)
env.reset()
print(env.observation_space(0))
print(env.observation_space(0)['TGT'].nvec)
# for i in range(30):
#     env.step([env.action_space(0).sample(),env.action_space(1).sample()])
#     env.render()
#     sleep(2)