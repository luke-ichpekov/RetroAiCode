import gym
import retro
import numpy as np
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import PPO2, A2C



class Discretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ["B", 'NONE', "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        actions = [["LEFT"], ["DOWN"], ["UP"], ["A"], ["B"]]
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
        #env = retro.make(env_id, state="1Player.Level1", scenario="scenario")
        env = Discretizer(retro.RetroEnv(env_id, state="1Player.Level1", scenario="scenario", obs_type=retro.Observations.RAM))
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "KungFu-Nes"
    num_cpu = 10
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2(MlpPolicy, env= env, verbose=1, learning_rate=0.00025, tensorboard_log="KungFuSadgeHealthPen")
    #model = PPO2.load('hellfire1milRam.zip', env)
    model.learn(1000000)
    #model.save("100MillionKungFu")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
