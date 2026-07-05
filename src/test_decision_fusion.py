from decision_fusion_engine import (
    fuse_decisions
)

result = fuse_decisions(

    traffic_signal="GREEN",

    risk_decision="MOVE",

    risk_speed=40,

    rl_action="MOVE",

    left_lane_status="UNSAFE",

    right_lane_status="CAUTION",

    lane_recommendation=
        "KEEP CURRENT LANE",

    collision_risk=150
)

print("\n======================")
print("DECISION FUSION TEST")
print("======================")

for k, v in result.items():

    print(k, ":", v)