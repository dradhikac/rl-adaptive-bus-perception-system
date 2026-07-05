import cv2

# ====================================
# PEDESTRIAN POSITION ANALYSIS
# ====================================

def analyze_pedestrians(
    detections,
    image_width,
    image_height
):

    pedestrian_count = 0

    in_lane_count = 0

    near_lane_count = 0

    sidewalk_count = 0

    pedestrian_status = "NONE"

    # ====================================
    # DRIVING LANE REGION
    # ====================================

    lane_left = image_width * 0.40
    lane_right = image_width * 0.60

    print("\n===== PEDESTRIAN DEBUG =====")

    for box in detections:

        class_name = box["class"]

        if class_name != "person":
            continue

        pedestrian_count += 1

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        center_x = (x1 + x2) / 2

        bottom_y = y2

        pedestrian_width = x2 - x1
        pedestrian_height = y2 - y1

        pedestrian_area = (
            pedestrian_width *
            pedestrian_height
        )

        print(
            f"Center={center_x:.0f} "
            f"Bottom={bottom_y:.0f} "
            f"Area={pedestrian_area:.0f}"
        )

        # ====================================
        # VERY CLOSE PEDESTRIAN
        # Must be:
        # 1. Inside lane
        # 2. Close to vehicle
        # 3. Large enough
        # ====================================

        if (

            lane_left <= center_x <= lane_right

            and

            bottom_y > image_height * 0.80

            and

            pedestrian_area > 25000

        ):

            in_lane_count += 1

        # ====================================
        # PEDESTRIAN NEAR LANE
        # ====================================

        elif (

            lane_left <= center_x <= lane_right

            and

            pedestrian_area > 10000

        ):

            near_lane_count += 1

        # ====================================
        # SIDEWALK / FAR PEDESTRIAN
        # ====================================

        else:

            sidewalk_count += 1

    # ====================================
    # STATUS
    # ====================================

    if in_lane_count > 0:

        pedestrian_status = "IN_LANE"

    elif near_lane_count > 0:

        pedestrian_status = "NEAR_LANE"

    elif sidewalk_count > 0:

        pedestrian_status = "SIDEWALK"

    else:

        pedestrian_status = "NONE"

    print("\n===== PEDESTRIAN ANALYSIS =====")

    print(
        "Pedestrians:",
        pedestrian_count
    )

    print(
        "In Lane:",
        in_lane_count
    )

    print(
        "Near Lane:",
        near_lane_count
    )

    print(
        "Sidewalk:",
        sidewalk_count
    )

    print(
        "Status:",
        pedestrian_status
    )

    print("==============================\n")

    return {

        "pedestrian_count":
            pedestrian_count,

        "in_lane_count":
            in_lane_count,

        "near_lane_count":
            near_lane_count,

        "sidewalk_count":
            sidewalk_count,

        "pedestrian_status":
            pedestrian_status
    }