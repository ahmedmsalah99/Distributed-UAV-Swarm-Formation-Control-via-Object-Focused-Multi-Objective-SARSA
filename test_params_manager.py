from common.parameters_manager import ParametersManager
# Example usage
if __name__ == "__main__":
    # Initialize the singleton parameters manager
    params_manager = ParametersManager("configs/train_phase1.yaml")

    # Access configuration parameters
    mode = params_manager.get_mode()
    env_params = params_manager.get_env_params()

    print("Mode:", mode)
    print("Environment Parameters:", env_params)
