# ====================================
# DECISION FUSION ENGINE
# ====================================

def fuse_decisions(

    traffic_signal,

    risk_decision,

    risk_speed,

    rl_action,

    left_lane_status,

    right_lane_status,

    lane_recommendation,

    collision_risk

):

    reasons = []

    # ====================================
    # ABSOLUTE SAFETY RULES
    # ====================================

    if traffic_signal == "RED":

        return {

            "final_decision":
                "STOP",

            "final_speed":
                0,

            "lane_action":
                "KEEP_LANE",

            "reason":
                "Red traffic signal"
        }

    # ====================================
    # LANE CHANGE ENGINE
    # ====================================

    lane_action = "KEEP_LANE"

    if rl_action == "CHANGE_LEFT":

        if left_lane_status == "SAFE":

            lane_action = "CHANGE_LEFT"

            reasons.append(
                "Left lane safe"
            )

        else:

            reasons.append(
                "Left lane occupied"
            )

    elif rl_action == "CHANGE_RIGHT":

        if right_lane_status == "SAFE":

            lane_action = "CHANGE_RIGHT"

            reasons.append(
                "Right lane safe"
            )

        else:

            reasons.append(
                "Right lane occupied"
            )

    else:

        lane_action = "KEEP_LANE"

    # ====================================
    # COLLISION SAFETY OVERRIDE
    # ====================================

    if collision_risk >= 150:

        lane_action = "KEEP_LANE"

        reasons.append(
            "High blind spot risk"
        )

    # ====================================
    # FORWARD DRIVING ENGINE
    # ====================================

    final_decision = risk_decision

    final_speed = risk_speed

    # ====================================
    # RL OVERRIDE
    # ====================================

    if risk_decision != "STOP":

        if rl_action == "STOP":

            final_decision = "STOP"

            final_speed = 0

            reasons.append(
                "RL emergency stop"
            )

        elif rl_action == "SLOW_DOWN":

            final_decision = "SLOW_DOWN"

            final_speed = min(
                risk_speed,
                20
            )

            reasons.append(
                "RL slowdown"
            )

        elif rl_action == "MOVE":

            reasons.append(
                "RL move"
            )

    # ====================================
    # OUTPUT
    # ====================================

    return {

        "final_decision":
            final_decision,

        "final_speed":
            final_speed,

        "lane_action":
            lane_action,

        "reason":
            ", ".join(reasons)
    }