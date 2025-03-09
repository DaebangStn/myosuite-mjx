from myosuite.utils import gym
import numpy as np
import os
import pickle

pth = "myosuite/agents/baslines_NPG/"
policy = (
    pth
    + "myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"
)
pi = pickle.load(open(policy, "rb"))

env = gym.make("myoElbowPose1D6MExoRandom-v0")

env.reset()

AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
env.reset()
frames = []
for ep in range(len(AngleSequence)):
    print(
        "Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep])
    )
    env.unwrapped.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]
    env.unwrapped.target_type = "fixed"
    env.unwrapped.weight_range = (0, 0)
    env.unwrapped.update_target()
    for _ in range(40):
        frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=0)
        frames.append(frame)
        o = env.get_obs()
        a = pi.get_action(o)[0]
        next_o, r, done, *_, ifo = env.step(
            a
        )  # take an action based on the current observation
env.close()
