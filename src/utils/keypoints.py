keypoint_name_to_idx_dict = {
    "left_brow_left_corner": (0, 1),
    "left_brow_right_corner": (2, 3),
    "right_brow_left_corner": (4, 5),
    "right_brow_right_corner": (6, 7),
    "left_eye_left_corner": (8, 9),
    "left_eye_pupil": (10, 11),
    "left_eye_right_corner": (12, 13),
    "right_eye_left_corner": (14, 15),
    "right_eye_pupil": (16, 17),
    "right_eye_right_corner": (18, 19),
    "nose": (20, 21),
    "mouth_left_corner": (22, 23),
    "mouth_middle": (24, 25),
    "mouth_right_corner": (26, 27),
}


def get_left_eye_middle_coords(keypoints: list[float]):

    l_idx = keypoint_name_to_idx_dict["left_eye_left_corner"]
    l_coords = (keypoints[l_idx[0]], keypoints[l_idx[1]])
    r_idx = keypoint_name_to_idx_dict["left_eye_right_corner"]
    r_coords = (keypoints[r_idx[0]], keypoints[r_idx[1]])
    eye_coords = zip(l_coords, r_coords)

    return [(c_r + c_l) / 2 for c_l, c_r in eye_coords]


def get_right_eye_middle_coords(keypoints: list[float]):

    l_idx = keypoint_name_to_idx_dict["right_eye_left_corner"]
    l_coords = (keypoints[l_idx[0]], keypoints[l_idx[1]])
    r_idx = keypoint_name_to_idx_dict["right_eye_right_corner"]
    r_coords = (keypoints[r_idx[0]], keypoints[r_idx[1]])
    eye_coords = zip(l_coords, r_coords)

    return [(c_r + c_l) / 2 for c_l, c_r in eye_coords]


complex_keypoints_to_coords_mapping = {
    "left_eye_middle": get_left_eye_middle_coords,
    "right_eye_middle": get_right_eye_middle_coords,
}
