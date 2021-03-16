import gym
import retro
import numpy as np
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import PPO2, A2C
from stable_baselines.common.callbacks import EvalCallback
import pickle




class Discretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ["B", 'NONE', "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        actions = [["LEFT"], ["DOWN"], ["UP"], ["RIGHT"]]
        self._actions = []
        for action in actions:
            arr = np.array([False]*12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))
    def action(self, a):
        return self._actions[a].copy()

def make_env(env_id, rank, seed=0):
    def _init():
        env = retro.RetroEnv(env_id, obs_type=retro.Observations.RAM)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "SamuraiShodown-Genesis"
    num_cpu = 10
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=50000, deterministic=True, render=False)
    #model = PPO2(MlpPolicy, env=env, verbose=1, learning_rate=0.003, cliprange_vf=-1, tensorboard_log="placeholder")
    model = PPO2.load('3milsamuraishowdown-nes.zip', env)
    #model.learn(3000000, tb_log_name="pacman1mil+run")
    #model.save("3milsamuraishowdown-nes")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render()
