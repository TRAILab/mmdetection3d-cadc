import json
import math
import os
from datetime import datetime

import mmengine
import numpy as np
import utm
import yaml
from dataset_converters.update_infos_to_v2 import (
    clear_instance_unused_keys, get_empty_instance,
    get_empty_standard_data_info)


def cadc_converter(root_path, info_prefix, out_dir):
    # TODO support camera data, currently only support lidar data
    infos = []
    for date in os.listdir(root_path):
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
                    calib_dict[f"intrinsics_{calib_file.split('.')[0]}"] = yaml.safe_load(f)

        for seq in os.listdir(date_path):
            if seq == 'calib':
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
            for i, (ann, lidar_ts, novatel_ts) in enumerate(zip(ann_seq, lidar_timestamps, novatel_timestamps)):
                ann['calib_dict'] = calib_dict
                ann['lidar_path'] = os.path.join(date, seq, 'labeled', 'lidar_points', 'data', str(i).zfill(10)+'.bin')
                if not os.path.exists(os.path.join(root_path, ann['lidar_path'])):
                    print(f"WARNING: lidar path {ann['lidar_path']} does not exist")
                ann['date'] = date
                ann['seq'] = seq
                ann['frame_idx'] = i
                # load novatel message
                novatel_path = os.path.join(seq_path, 'labeled', 'novatel', 'data', str(i).zfill(10)+'.txt')
                with open(novatel_path, 'r') as f:
                    novatel_data = f.readline().split(' ')
                if origin is None:
                    origin = utm.from_latlon(float(novatel_data[0]), float(novatel_data[1]))
                    origin = [origin[0], origin[1], float(novatel_data[2]) + float(novatel_data[3])]
                ann['pose'] = novatel2pose(novatel_data, origin)
                # drop last 3 digits, datetime only supports microsecond precision, and the \n char
                ann['lidar_timestamp'] = datetime.strptime(lidar_ts[:-4], "%Y-%m-%d %H:%M:%S.%f")
                ann['novatel_timestamp'] = datetime.strptime(novatel_ts[:-4], "%Y-%m-%d %H:%M:%S.%f")
                infos.append(ann)
    # update ann infos to v2
    converted_list = []
    camera_types = [
        'camera_F',
        'camera_FR',
        'camera_RF',
        'camera_RB',
        'camera_B',
        'camera_LB',
        'camera_LF',
        'camera_FL',
    ]
    for i, ori_info_dict in enumerate(mmengine.track_iter_progress(infos)):
        temp_data_info = get_empty_standard_data_info(camera_types=camera_types)
        temp_data_info['sample_idx'] = i
        temp_data_info['timestamp_s'] = ori_info_dict['lidar_timestamp'].timestamp()
        temp_data_info['ego2global'] = ori_info_dict['pose']
        temp_data_info['lidar_points']['lidar2ego'] = ori_info_dict['calib_dict']['extrinsics']['T_LIDAR_GPSIMU']
        temp_data_info['lidar_points']['lidar_path'] = ori_info_dict['lidar_path']
        temp_data_info['lidar_points']['num_pts_feats'] = 4
        # gt box infos
        for i, gt_dict in enumerate(ori_info_dict['cuboids']):
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
    metainfo['version'] = 'full' # all 3 scenes, no train/val split
    metainfo['info_version'] = '2.0'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)
    mmengine.dump(converted_data_info, os.path.join(out_dir, f'{info_prefix}_infos_v2.pkl'))


def novatel2pose(gps_msg, origin):
    # utm_data[0] = East (m), utm_data[1] = North (m)
    utm_data = utm.from_latlon(float(gps_msg[0]), float(gps_msg[1]));
    # Ellipsoidal height = MSL (orthometric) + undulation
    ellipsoidal_height = float(gps_msg[2]) + float(gps_msg[3]);

    roll = np.deg2rad(float(gps_msg[7]));
    pitch = np.deg2rad(float(gps_msg[8]));

    # Azimuth = north at 0 degrees, east at 90 degrees, south at 180 degrees and west at 270 degrees
    azimuth = float(gps_msg[9]);
    # yaw = north at 0 deg, 90 at west and 180 at south, east at 270 deg
    yaw = np.deg2rad(-1.0 * azimuth); 

    c_phi = math.cos(roll);
    s_phi = math.sin(roll);
    c_theta = math.cos(pitch);
    s_theta = math.sin(pitch);
    c_psi = math.cos(yaw);
    s_psi = math.sin(yaw);

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

    return pose;
