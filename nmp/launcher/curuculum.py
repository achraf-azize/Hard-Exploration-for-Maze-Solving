import os

from nmp.launcher.sac import sac
import gtimer as gt
import gym
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed, setup_logger
from nmp.policy import utils as utils_pol
from rlkit.samplers.rollout_functions import multitask_rollout


class Curriculum:
    def __init__(self, var, grid_size, success_to_next=0.7, max_epochs=1500, epochs_per_curu=20, curu_dist_step=0.2,
                 curu_obstacles_step=3, cpu=False, resume_from=None):
        self.variant = var.copy()
        self.grid_size = grid_size
        self.max_epochs = max_epochs
        self.cpu = cpu
        self.epochs_per_curu = epochs_per_curu
        self.success_to_next = success_to_next
        self.curu_dist_step = curu_dist_step
        self.curu_obstacles_step = curu_obstacles_step
        self.variant['algorithm_kwargs']['num_epochs'] = epochs_per_curu
        self.seed = var['seed']
        self.env_name = var['env_name']
        self.horizon = var['algorithm_kwargs']['max_path_length']
        self.log_dir_base = var["log_dir"]
        self.last_dir = None
        self.last_experiment_dir = resume_from

    def create_variant_curu(self, env_kwargs):
        variant = self.variant.copy()
        variant['env_kwargs'] = env_kwargs
        self.env_kwargs = env_kwargs
        variant['log_dir'] = os.path.join(self.log_dir_base, f"grid_size_{env_kwargs['grid_size']}",
                                          f"distance_start_goal_{env_kwargs['distance_start_goal']}",
                                          f"n_obstacles_{env_kwargs['n_obstacles']}")
        return variant

    def evaluate(self, stochastic=False, episodes=100):
        set_seed(self.seed)
        env = gym.make(self.env_name, **self.env_kwargs)
        env.seed(self.seed)
        env.set_eval()
        policy = utils_pol.load(self.last_dir, "itr_0.pkl", self.cpu, stochastic)
        #policy = utils_pol.load("/home/achraf/Downloads/", "params.pkl", self.cpu, stochastic)
        if stochastic:
            num_params = policy.num_params()
        else:
            num_params = policy.stochastic_policy.num_params()
        print(f"num params: {num_params}")

        horizon = self.horizon
        reset_kwargs = {}

        def rollout_fn():
            return multitask_rollout(
                env,
                policy,
                horizon,
                False,
                observation_key="observation",
                desired_goal_key="desired_goal",
                representation_goal_key="representation_goal",
                **reset_kwargs,
            )

        success_rate, n_col, paths_states = utils_pol.evaluate(rollout_fn, episodes)
        print(f"Success rate: {success_rate} - Collisions: {n_col}")
        return success_rate

    def train_one_curu(self, env_kwargs, j):
        variant = self.create_variant_curu(env_kwargs)
        exp_dir = variant['log_dir']
        setup_logger_kwargs = {
            "exp_prefix": exp_dir,
            "variant": variant,
            "log_dir": exp_dir,
            "snapshot_mode": "all",
            "snapshot_gap": 1,
        }
        setup_logger(**setup_logger_kwargs)
        ptu.set_gpu_mode(not self.cpu, distributed_mode=False)
        print(f"Start training...")
        gt.reset_root()
        print(self.last_experiment_dir)
        sac(variant, cpu=self.cpu, resume_from=self.last_experiment_dir)
        self.last_dir = exp_dir
        self.last_experiment_dir = os.path.join(exp_dir, f"itr_{j}.pkl")

    def train(self):
        env_kwargs = dict(grid_size=self.grid_size,
                          n_obstacles=0,
                          distance_start_goal=0.2)
        j = self.epochs_per_curu - 1
        self.train_one_curu(env_kwargs, j)
        n = 1
        p = self.evaluate()
        while n < self.max_epochs / self.epochs_per_curu:
            while p < self.success_to_next and n < self.max_epochs / self.epochs_per_curu:
                self.train_one_curu(env_kwargs, j)
                n += 1
                p = self.evaluate()
            env_kwargs['n_obstacles'] += self.curu_obstacles_step
            print('NEW STAGE WITH n_obstacle:', env_kwargs['n_obstacles'])
            if env_kwargs['n_obstacles'] > (self.grid_size - 1) ** 2:
                env_kwargs['n_obstacles'] = 0
                env_kwargs['distance_start_goal'] += self.curu_dist_step
                print('NEW STAGE WITH n_obstacle and dist:', env_kwargs['n_obstacles'], env_kwargs['distance_start_goal'])
                if env_kwargs['distance_start_goal'] > 0.8:
                    return
            if n < self.max_epochs / self.epochs_per_curu:
                self.train_one_curu(env_kwargs, j)
                n += 1
                p = self.evaluate()