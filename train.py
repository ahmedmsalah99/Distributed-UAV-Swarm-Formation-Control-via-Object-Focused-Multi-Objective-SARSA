from uav_env.env.uav_environment import UAVEnvironment
from time import sleep
from common.utils import Point
from common.parameters_manager import ParametersManager
from modules.modules_manager import TargetSeekModule,CohesionModule,CollisionModule,AlignmentModule,ObstacleAvoidanceModule
from common.sarsa_learning import SARSALearning


parameters_manager = ParametersManager('configs/train_phase1.yaml')

## Modules Initialization
modules_configs = parameters_manager.get_active_modules()
training_params = parameters_manager.get_training_params()

modules = {}
Qmatrix_path_dict = {}
for mod in modules_configs:

    config = modules_configs[mod]
    if "Qmatrix_path" in config:
            Qmatrix_path_dict[mod] = config["Qmatrix_path"]

    if mod == "TGT":
        modules[mod] = TargetSeekModule(config['w'],config['discount'],config['thresh'])
    elif mod == "COH":
        modules[mod] = CohesionModule(config['w'],config['discount'])
    elif mod == "COL":
        modules[mod] = CollisionModule(config['w'],config['discount'],config['away_thresh'],config['close_thresh'])
    elif mod == "ALN":
        modules[mod] = AlignmentModule(config['w'],config['discount'])
    elif mod == "OBS":
        modules[mod] = ObstacleAvoidanceModule(config['w'],config['discount'],config['away_thresh'],config['close_thresh'])

env_params = parameters_manager.get_env_params()
env = UAVEnvironment(agents_num=env_params['agents_num'],env_params=env_params,modules=modules)
env.reset()


mode = parameters_manager.get_mode()
training_mode = mode == "train"
learning_module = SARSALearning(env,modules,training_params,training_mode=mode,Qmatrix_path_dict=Qmatrix_path_dict)
learning_module.simulate_episodes()

