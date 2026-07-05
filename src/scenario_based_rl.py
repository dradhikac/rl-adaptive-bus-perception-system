import pandas as pd
import random

# ====================================
# ACTIONS
# ====================================

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP",

    "CHANGE_LEFT",

    "CHANGE_RIGHT"
]

# ====================================
# RESEARCH-LEVEL SCENARIOS
# ====================================

scenarios = [

    {
        "name": "Clear Road",

        "state": (
            "GREEN",
            "FAR",
            "LOW",
            "CLEAR",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "MOVE"
    },

    {
        "name": "Vehicle Ahead",

        "state": (
            "GREEN",
            "MEDIUM",
            "LOW",
            "CLEAR",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "SLOW_DOWN"
    },

    {
        "name": "Red Signal",

        "state": (
            "RED",
            "VERY_CLOSE",
            "HIGH",
            "CLEAR",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "STOP"
    },

    {
        "name": "Pedestrian Crossing",

        "state": (
            "GREEN",
            "FAR",
            "MEDIUM",
            "PEDESTRIAN",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "SLOW_DOWN"
    },

    {
        "name": "Crowded Crosswalk",

        "state": (
            "GREEN",
            "FAR",
            "HIGH",
            "CROWDED",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "STOP"
    },
    {
        "name": "Pedestrian Ahead",

        "state": (
            "GREEN",
            "FAR",
            "HIGH",
            "PEDESTRIAN",
            "SAFE",
            "SAFE"
        ),

        "best_action":
            "STOP"
    },

    {
        "name": "Right Occupied",

        "state": (
            "GREEN",
            "MEDIUM",
            "LOW",
            "CLEAR",
            "SAFE",
            "UNSAFE"
        ),

        "best_action":
            "CHANGE_LEFT"
    },

    {
        "name": "Left Occupied",

        "state": (
            "GREEN",
            "MEDIUM",
            "LOW",
            "CLEAR",
            "UNSAFE",
            "SAFE"
        ),

        "best_action":
            "CHANGE_RIGHT"
    },

    {
        "name": "Both Occupied",

        "state": (
            "GREEN",
            "MEDIUM",
            "HIGH",
            "CLEAR",
            "UNSAFE",
            "UNSAFE"
        ),

        "best_action":
            "STOP"
    },

    {
        "name": "Crowded Crosswalk Left Unsafe",

        "state": (
            "GREEN",
            "FAR",
            "HIGH",
            "CROWDED",
            "UNSAFE",
            "SAFE"
        ),

        "best_action":
            "STOP"
    },

    {
        "name": "Crowded Crosswalk Right Unsafe",

        "state": (
            "GREEN",
            "FAR",
            "HIGH",
            "CROWDED",
            "SAFE",
            "UNSAFE"
        ),

        "best_action":
            "STOP"
    }
]

# ====================================
# Q TABLE
# ====================================

q_table = {}

for scenario in scenarios:

    state = scenario["state"]

    q_table[state] = {

        action: 0.0

        for action in actions
    }

# ====================================
# REWARD FUNCTION
# ====================================

def get_reward(

    scenario,

    action

):

    if action == scenario["best_action"]:

        return 100

    return -50

# ====================================
# TRAINING
# ====================================

alpha = 0.1

episodes = 10000

print("\nTraining Scenario RL Agent...\n")

for episode in range(episodes):

    scenario = random.choice(
        scenarios
    )

    state = scenario["state"]

    action = random.choice(
        actions
    )

    reward = get_reward(
        scenario,
        action
    )

    old_q = q_table[state][action]

    new_q = old_q + alpha * (

        reward - old_q

    )

    q_table[state][action] = new_q

# ====================================
# DISPLAY POLICY
# ====================================

print("=" * 50)
print(" RESEARCH LEVEL RL POLICY ")
print("=" * 50)

policy_rows = []

for scenario in scenarios:

    state = scenario["state"]

    best_action = max(
        q_table[state],
        key=q_table[state].get
    )

    print("\nScenario:")

    print(
        scenario["name"]
    )

    print(
        "State:",
        state
    )

    print(
        "Learned Action:",
        best_action
    )

    policy_rows.append({

        "scenario":
            scenario["name"],

        "traffic_signal":
            state[0],

        "vehicle_distance":
            state[1],

        "risk_level":
            state[2],

        "pedestrian_zone":
            state[3],

        "blind_left":
            state[4],

        "blind_right":
            state[5],

        "best_action":
            best_action
    })

# ====================================
# SAVE POLICY
# ====================================

df = pd.DataFrame(
    policy_rows
)

df.to_csv(

    "outputs/scenario_rl_policy.csv",

    index=False
)

print("\nSaved:")
print(
    "outputs/scenario_rl_policy.csv"
)