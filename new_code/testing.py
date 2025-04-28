
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from overcooked_ai_py.mdp.actions import Action

# Create an OvercookedGridworld layout
mdp = OvercookedGridworld.from_layout_name("cramped_room")

state = mdp.get_standard_start_state()

print(type(mdp))

# Use the correct method to create the environment
env = OvercookedEnv.from_mdp(mdp, horizon=400)

# Now you're good to go!
obs = env.reset()
done = False

print(Action.ALL_ACTIONS)
actions = Action.ALL_ACTIONS

while not done:
    action_0 = actions[0]
    action_1 = actions[0]
    print(obs)
    obs, reward, done, info = env.step((action_0, action_1))
    print("Reward:", reward)

    print(len(obs.to_dict()))