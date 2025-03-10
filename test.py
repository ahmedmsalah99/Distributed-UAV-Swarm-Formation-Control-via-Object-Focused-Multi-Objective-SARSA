from uav_env.env.uav_environment import UAVEnvironment

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = UAVEnvironment(agents_num=5)
    parallel_api_test(env, num_cycles=1_000_000)