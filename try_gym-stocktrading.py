
import time
import gym
# from stable_baselines import PPO2
# from stable_baselines.common.policies import MlpPolicy

env = gym.make('gym_stocktrading:stocktrading-v0')
obs = env.reset()
done = False

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000)

for i in range(100):
    while not done:
        action = env.action_space.sample()
        # action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        time.sleep(0.5)
        env.render(mode='text')

        if done:
            ob = env.reset()
            done = False

env.close()