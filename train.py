from uav_env.env.uav_environment import UAVEnvironment
from time import sleep
from common.utils import Point
from common.parameters_manager import ParametersManager
from modules.modules_manager import TargetSeekModule,CohesionModule,CollisionModule,AlignmentModule,ObstacleAvoidanceModule
from common.sarsa_learning import SARSALearning


parameters_manager = ParametersManager('configs/train_phase1.yaml')

## Modules Initialization
modules_configs = parameters_manager.get_active_modules()
modules = {}
for mod in modules_configs:
    if mod == "TGT":
        config = modules_configs[mod]
        modules[mod] = TargetSeekModule(config['w'],config['discount'],config['thresh'])
    elif mod == "COH":
        config = modules_configs[mod]
        modules[mod] = CohesionModule(config['w'],config['discount'])
    elif mod == "COL":
        config = modules_configs[mod]
        modules[mod] = CollisionModule(config['w'],config['discount'])
    elif mod == "ALN":
        config = modules_configs[mod]
        modules[mod] = AlignmentModule(config['w'],config['discount'])
    elif mod == "OBS":
        config = modules_configs[mod]
        modules[mod] = ObstacleAvoidanceModule(config['w'],config['discount'])

env_params = parameters_manager.get_env_params()
env = UAVEnvironment(agents_num=env_params['agents_num'],obstacles_num=env_params['obstacles_num'],
                     waypoints_num=env_params['waypoints_num'],size=Point(10,10),modules=modules,
                     update_waypoint=env_params['update_waypoint'],max_dist=10)
env.reset()
training_params = parameters_manager.get_training_params()
learning_module = SARSALearning(env,modules,training_params['alpha'],training_params['T'],training_params['number_episodes'],Qmatrix_path_dict = {"TGT":"Qmatrix_14999.npy"},training_mode=False)
learning_module.simulate_episodes()

# for i in range(30):
#     env.step([env.action_space(0).sample(),env.action_space(1).sample()])
#     env.render()
#     sleep(2)