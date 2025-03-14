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
env = UAVEnvironment(agents_num=env_params['agents_num'],obstacles_num=env_params['obstacles_num'],
                     waypoints_num=env_params['waypoints_num'],size=Point(env_params["x_size"],env_params["y_size"]),modules=modules,
                     update_waypoint=env_params['update_waypoint'],max_dist=env_params["max_dist"],termination_reward =training_params['termination_reward'],
                     waypoint_dist_thesh=env_params['waypoint_dist_thesh'],timeout = training_params["timeout"],
                     waypoint_in_radius = env_params["waypoint_in_radius"],spawn_center=Point(env_params['spawn_center_x'],env_params['spawn_center_y']))
env.reset()
# Qmatrix_path_dict = {"TGT":'./phase1/Qmatrix_9999_good_shaky_3.npy'}
# Qmatrix_path_dict = { mod:"Qmatrix_1999_good.npy"
#     for mod in ["COH","COL","ALN","OBS"]
# }
# Qmatrix_path_dict = { mod:"phase2/Qmatrix_999.npy"
#     for mod in ["COH","COL","ALN","OBS"]
# }

mode = parameters_manager.get_mode()
training_mode = mode == "train"
learning_module = SARSALearning(env,modules,training_params['alpha'],training_params['T'],training_params['number_episodes'],
                                save_folder=training_params["save_folder"],
                                training_mode= training_mode,train_plot_title=training_params["train_plot_title"]
                                ,window_size=training_params["window_size"])#,Qmatrix_path_dict=Qmatrix_path_dict)
learning_module.simulate_episodes()

