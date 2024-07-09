import json
import math
import os
from datetime import datetime, timedelta

import mmengine
import numpy as np
import utm
import yaml
from dataset_converters.update_infos_to_v2 import (
    clear_instance_unused_keys, get_empty_instance,
    get_empty_standard_data_info, get_single_lidar_sweep)

CAMERA_TYPES = [
    'camera_F',
    'camera_FR',
    'camera_RF',
    'camera_RB',
    'camera_B',
    'camera_LB',
    'camera_LF',
    'camera_FL',
]

# val scenes from
# https://github.com/mpitropov/OpenPCDet/blob/cadc_support/data/cadc/ImageSets/generate_splits.py
# val_set = [
#     ['2018_03_06', '0001'],
#     ['2018_03_06', '0008'],
#     ['2018_03_06', '0016'],
#     ['2018_03_07', '0004'],
#     ['2019_02_27', '0009'],
#     ['2019_02_27', '0016'],
#     ['2019_02_27', '0028'],
#     ['2019_02_27', '0033'],
#     ['2019_02_27', '0040'],
#     ['2019_02_27', '0043'],
#     ['2019_02_27', '0054'],
#     ['2019_02_27', '0060'],
#     ['2019_02_27', '0065'],
#     ['2019_02_27', '0076'],
# ]

# bugged scenes
# ['2019_02_27', '0004'] # gt appears to have some rotational offset


def cadc_converter(root_path, trainval_json: str, info_prefix, out_dir, max_sweeps: int = 10):
    # TODO support camera data, currently only support lidar data
    # read trainval json
    with open(trainval_json, 'r') as f:
        trainval_split = json.load(f)
    trainval_split = {x['seq']: x['split']
                      for x in trainval_split if 'cam0' in x['seq']}
    # drop the camera directory to just keep the splits
    # also only take based on cam0 splits
    trainval_split = {os.path.dirname(k): v for k, v in trainval_split.items()}
    all_infos = []
    train_infos = []
    val_infos = []
    obj_db = {}
    # iterate through each date
    for date in os.listdir(root_path):
        if 'gt_database' in date:
            continue
        date_path = os.path.join(root_path, date)
        if not os.path.isdir(date_path):
            continue
        print("Begin processing date: ", date)
        calib_path = os.path.join(date_path, 'calib')
        calib_dict = {}
        for calib_file in os.listdir(calib_path):
            calib_file_path = os.path.join(calib_path, calib_file)
            if "README" in calib_file:
                continue
            elif "extrinsics" in calib_file:
                with open(calib_file_path, 'r') as f:
                    calib_dict['extrinsics'] = yaml.safe_load(f)
            else:
                with open(calib_file_path, 'r') as f:
                    calib_dict[f"intrinsics_{calib_file.split('.')[0]}"] = yaml.safe_load(
                        f)
        T_lidar_imu = np.array(
            calib_dict['extrinsics']['T_LIDAR_GPSIMU'])
        T_imu_lidar = np.linalg.inv(T_lidar_imu)
        # iterate through each clip in the date
        for seq in os.listdir(date_path):
            if seq == 'calib':
                continue
            if os.path.join(date, seq) not in trainval_split:
                print(
                    f"{os.path.join(date, seq)} was not found in the trainval_json. Not including in pkls")
                continue
            seq_path = os.path.join(date_path, seq)
            ann_path = os.path.join(seq_path, '3d_ann.json')
            with open(ann_path, 'r') as f:
                ann_seq = json.load(f)
            origin = None
            with open(os.path.join(seq_path, 'labeled/lidar_points/timestamps.txt'), 'r') as f:
                lidar_timestamps = f.readlines()
            with open(os.path.join(seq_path, 'labeled/novatel/timestamps.txt'), 'r') as f:
                novatel_timestamps = f.readlines()
            # iterate through each frame in the clip
            for i, (ann, lidar_ts, novatel_ts) in enumerate(zip(ann_seq, lidar_timestamps, novatel_timestamps)):
                ann['calib_dict'] = calib_dict
                ann['lidar_path'] = os.path.join(
                    date, seq, 'labeled', 'lidar_points', 'data', str(i).zfill(10)+'.bin')
                if not os.path.exists(os.path.join(root_path, ann['lidar_path'])):
                    print(
                        f"WARNING: lidar path {ann['lidar_path']} does not exist")
                ann['date'] = date
                ann['seq'] = seq
                ann['frame_idx'] = i
                sweeps_list = []
                # load novatel message
                novatel_path = os.path.join(
                    seq_path, 'labeled', 'novatel', 'data', str(i).zfill(10)+'.txt')
                with open(novatel_path, 'r') as f:
                    novatel_data = f.readline().split(' ')
                if origin is None:
                    origin = utm.from_latlon(
                        float(novatel_data[0]), float(novatel_data[1]))
                    origin = [origin[0], origin[1], float(
                        novatel_data[2]) + float(novatel_data[3])]
                # drop last 3 digits, datetime only supports microsecond precision, and the \n char
                ann['lidar_timestamp'] = datetime.strptime(
                    lidar_ts[:-4], "%Y-%m-%d %H:%M:%S.%f")
                ann['novatel_timestamp'] = datetime.strptime(
                    novatel_ts[:-4], "%Y-%m-%d %H:%M:%S.%f")
                T_global_imu = novatel2pose(novatel_data, origin)
                ann['pose'] = T_global_imu
                T_global_lidar = np.dot(T_global_imu, T_imu_lidar)
                # extract velocity from pose info
                for obj in ann['cuboids']:
                    if obj['uuid'] not in obj_db:
                        # initialize entry in obj_db
                        obj_db[obj['uuid']] = {
                            'poses': np.empty((len(ann_seq), 3)),
                            'visible': np.zeros(len(ann_seq), dtype=bool),
                            'timestamps': np.empty(len(ann_seq), dtype=object)
                        }
                        obj_db[obj['uuid']]['poses'].fill(np.nan)
                        obj_db[obj['uuid']]['timestamps'].fill(np.nan)
                    # transform position to global frame
                    pos = [obj['position']['x'], obj['position']
                           ['y'], obj['position']['z']]
                    obj_db[obj['uuid']]['poses'][i] = np.dot(
                        T_global_lidar[:3, :3], pos) + T_global_lidar[:3, 3]
                    obj_db[obj['uuid']]['visible'][i] = True
                    obj_db[obj['uuid']]['timestamps'][i] = ann['lidar_timestamp']
                # load multiple sweeps
                for j in range(1, max_sweeps):
                    # check if prev sweep exists
                    if i-j < 0:
                        break
                    data_path = os.path.join(
                        date, seq, 'labeled', 'lidar_points', 'data', str(i-j).zfill(10)+'.bin')
                    sweep_j = all_infos[-j]
                    assert sweep_j['date'] == date and sweep_j['seq'] == seq and sweep_j['frame_idx'] == i-j
                    T_lidar_global = np.linalg.inv(T_global_lidar)
                    T_global_imu_prev = sweep_j['pose']
                    # sensor refers to prev sensor location. T_imu_lidar is a static matrix
                    T_global_sensor = T_global_imu_prev @ T_imu_lidar
                    T_lidar_sensor = T_lidar_global @ T_global_sensor
                    sweep = {
                        'data_path': data_path,
                        'type': 'lidar',
                        # 'sensor2ego': T_lidar_imu,
                        # 'ego2global': T_global_imu_prev,
                        'timestamp': datetime.strptime(lidar_timestamps[i-j][:-4], "%Y-%m-%d %H:%M:%S.%f"),
                        'sensor2lidar': T_lidar_sensor,
                    }
                    sweeps_list.append(sweep)
                ann['sweeps'] = sweeps_list
                all_infos.append(ann)
                if trainval_split[os.path.join(date, seq)] == 'train':
                    train_infos.append(ann)
                elif trainval_split[os.path.join(date, seq)] == 'val':
                    val_infos.append(ann)
    print("Finished processing all dates. Generating velocities")
    # generate velocities in global space
    for obj_uuid, obj_db_single in obj_db.items():
        obj_db[obj_uuid]['velocity'] = box_velocity(obj_db_single)
    # update ann infos to v2
    # TODO remove redundant processing from doing the split before the final conversion
    print("begin final conversion")
    generate_v2_pkl(info_prefix+'_all', out_dir, all_infos, obj_db)
    generate_v2_pkl(info_prefix+'_train', out_dir, train_infos, obj_db)
    generate_v2_pkl(info_prefix+'_val', out_dir, val_infos, obj_db)


def generate_v2_pkl(info_prefix, out_dir, infos, obj_db):
    converted_list = []
    for i, ori_info_dict in enumerate(mmengine.track_iter_progress(infos)):
        temp_data_info = get_empty_standard_data_info(
            camera_types=CAMERA_TYPES)
        temp_data_info['sample_idx'] = i
        temp_data_info['scene_token'] = os.path.join(
            ori_info_dict['date'], ori_info_dict['seq'])
        temp_data_info['token'] = os.path.join(
            ori_info_dict['date'], ori_info_dict['seq'], str(ori_info_dict['frame_idx']).zfill(10))
        temp_data_info['timestamp'] = ori_info_dict['lidar_timestamp'].timestamp()
        temp_data_info['ego2global'] = ori_info_dict['pose']
        temp_data_info['lidar_points']['lidar2ego'] = ori_info_dict['calib_dict']['extrinsics']['T_LIDAR_GPSIMU']
        temp_data_info['lidar_points']['lidar_path'] = ori_info_dict['lidar_path']
        temp_data_info['lidar_points']['num_pts_feats'] = 4
        # load multi-sweeps
        for ori_sweep in ori_info_dict['sweeps']:
            temp_lidar_sweep = get_single_lidar_sweep()
            # temp_lidar_sweep['lidar_points']['lidar2ego'] = ori_sweep['sensor2ego']
            # temp_lidar_sweep['ego2global'] = ori_sweep['ego2global']
            temp_lidar_sweep['lidar_points']['lidar2sensor'] = np.linalg.inv(
                ori_sweep['sensor2lidar'])
            temp_lidar_sweep['timestamp'] = ori_sweep['timestamp'].timestamp()
            temp_lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            # temp_lidar_sweep['sample_data_token'] = ori_sweep[
            #     'sample_data_token']
            # gt box infos
            temp_data_info['lidar_sweeps'].append(temp_lidar_sweep)
        for _, gt_dict in enumerate(ori_info_dict['cuboids']):
            empty_instance = get_empty_instance()
            # bbox_3d x y z l w h yaw
            # switch the dimensions x and y.
            # CADC has dimsension-x as (left-right)
            # mmdet3d has dimension-x as (front-back)
            empty_instance['bbox_3d'] = [
                gt_dict['position']['x'],
                gt_dict['position']['y'],
                gt_dict['position']['z'],
                gt_dict['dimensions']['y'],
                gt_dict['dimensions']['x'],
                gt_dict['dimensions']['z'],
                gt_dict['yaw']
            ]
            # velocity
            global_vel = obj_db[gt_dict['uuid']
                                ]['velocity'][ori_info_dict['frame_idx']]
            # transform velocity to ego frame
            t_global_imu = ori_info_dict['pose']
            t_imu_global = np.linalg.inv(t_global_imu)
            t_lidar_global = ori_info_dict['calib_dict']['extrinsics']['T_LIDAR_GPSIMU'] @ t_imu_global
            ego_vel = np.dot(t_lidar_global[:3, :3], global_vel)
            empty_instance['velocity'] = ego_vel[:2]

            # label
            empty_instance['bbox_label_3d'] = gt_dict['label']
            empty_instance['num_lidar_pts'] = gt_dict['points_count']
            # tracking ID
            empty_instance['instance_id'] = gt_dict['uuid']
            empty_instance = clear_instance_unused_keys(empty_instance)
            temp_data_info['instances'].append(empty_instance)
        converted_list.append(temp_data_info)
    metainfo = dict()
    metainfo['dataset'] = 'cadc'
    metainfo['version'] = 'full'  # all 3 scenes, no train/val split
    metainfo['info_version'] = '2.0'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)
    save_path = os.path.join(out_dir, f'{info_prefix}_infos_v2.pkl')
    mmengine.dump(converted_data_info, save_path)
    print("Finished converting to v2. Saved to ", save_path)


def novatel2pose(gps_msg, origin):
    # utm_data[0] = East (m), utm_data[1] = North (m)
    utm_data = utm.from_latlon(float(gps_msg[0]), float(gps_msg[1]))
    # Ellipsoidal height = MSL (orthometric) + undulation
    ellipsoidal_height = float(gps_msg[2]) + float(gps_msg[3])

    roll = np.deg2rad(float(gps_msg[7]))
    pitch = np.deg2rad(float(gps_msg[8]))

    # Azimuth = north at 0 degrees, east at 90 degrees, south at 180 degrees and west at 270 degrees
    azimuth = float(gps_msg[9])
    # yaw = north at 0 deg, 90 at west and 180 at south, east at 270 deg
    yaw = np.deg2rad(-1.0 * azimuth)

    c_phi = math.cos(roll)
    s_phi = math.sin(roll)
    c_theta = math.cos(pitch)
    s_theta = math.sin(pitch)
    c_psi = math.cos(yaw)
    s_psi = math.sin(yaw)

    # This is the T_locallevel_body transform where ENU is the local level frame
    # and the imu is the body frame
    # https://www.novatel.com/assets/Documents/Bulletins/apn037.pdf
    # pose = (np.matrix([
    #   [c_psi * c_phi - s_psi * s_theta * s_phi, -s_psi * c_theta, c_psi * s_phi + s_psi * s_theta * c_phi, utm_data[0] - origin[0]],
    #   [s_psi * c_phi + c_psi * s_theta * s_phi, c_psi * c_theta, s_psi * s_phi - c_psi * s_theta * c_phi, utm_data[1] - origin[1]],
    #   [-c_theta * s_phi, s_theta, c_theta * c_phi, ellipsoidal_height - origin[2]],
    #   [0.0, 0.0, 0.0, 1.0]]));
    pose = np.eye(4)
    pose[0, 0] = c_psi * c_phi - s_psi * s_theta * s_phi
    pose[0, 1] = -s_psi * c_theta
    pose[0, 2] = c_psi * s_phi + s_psi * s_theta * c_phi
    pose[0, 3] = utm_data[0] - origin[0]
    pose[1, 0] = s_psi * c_phi + c_psi * s_theta * s_phi
    pose[1, 1] = c_psi * c_theta
    pose[1, 2] = s_psi * s_phi - c_psi * s_theta * c_phi
    pose[1, 3] = utm_data[1] - origin[1]
    pose[2, 0] = -c_theta * s_phi
    pose[2, 1] = s_theta
    pose[2, 2] = c_theta * c_phi
    pose[2, 3] = ellipsoidal_height - origin[2]

    return pose


def box_velocity(obj_db_single, max_time_diff: float = 1.5) -> np.ndarray:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to np.nan.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
    """
    n = len(obj_db_single['visible'])
    ret = np.zeros((n, 3))
    ret.fill(np.nan)
    # only a single annotation, cannot compute velocity
    if obj_db_single['visible'].sum() == 1:
        return ret
    prevs = np.where(obj_db_single['visible'])[0][:-1]
    # first element has no prev, so it is its own prev
    prevs = np.insert(prevs, 0, prevs[0])
    nexts = np.where(obj_db_single['visible'])[0][1:]
    # last element has no next, so it is its own next
    nexts = np.append(nexts, nexts[-1])
    max_time_diff_td = timedelta(seconds=max_time_diff)
    for i, (prev, next) in enumerate(zip(prevs, nexts)):
        if not obj_db_single['visible'][i]:
            continue
        ts_next = obj_db_single['timestamps'][next]
        ts_prev = obj_db_single['timestamps'][prev]
        ts_curr = obj_db_single['timestamps'][i]
        if ts_next - ts_curr > max_time_diff_td:
            next = i
        if ts_curr - ts_prev > max_time_diff_td:
            prev = i
        if next == prev:
            continue

        ret[i] = (obj_db_single['poses'][next] - obj_db_single['poses'][prev]) / \
            ((obj_db_single['timestamps'][next] -
             obj_db_single['timestamps'][prev]).microseconds / 1e6)

    return ret
