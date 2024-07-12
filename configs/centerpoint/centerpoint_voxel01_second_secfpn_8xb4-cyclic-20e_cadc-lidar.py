_base_ = [
    '../_base_/datasets/cadc-lidar.py',
    '../_base_/models/centerpoint_voxel01_second_secfpn_nus.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car',
    'Truck',
    'Bus',
    'Traffic_Guidance_Objects',
    'Bicycle',
    'Pedestrian',
    'Pedestrian_With_Object',
    'Garbage_Containers_on_Wheels',
    # 'Horse_and_Buggy',
    # 'Animals'
]
data_prefix = dict(pts='', img='', sweeps='')
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(
        bbox_coder=dict(pc_range=point_cloud_range[:2]),
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Truck']),
            dict(num_class=1, class_names=['Bus']),
            dict(num_class=1, class_names=['Traffic_Guidance_Objects']),
            dict(num_class=1, class_names=['Bicycle']),
            dict(num_class=3, class_names=[
                 'Pedestrian', 'Pedestrian_With_Object', 'Garbage_Containers_on_Wheels']),
        ]),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'CADCDataset'
data_root = 'data/cadcd/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'cadc_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5,
            Truck=5,
            Bus=5,
            Traffic_Guidance_Objects=5,
            Bicycle=5,
            Pedestrian=5,
            Pedestrian_With_Object=5,
            Garbage_Containers_on_Wheels=5)),
    classes=class_names,
    sample_groups=dict(
        Car=2,
        Truck=3,
        Bus=4,
        Traffic_Guidance_Objects=2,
        Bicycle=6,
        Pedestrian=2,
        Pedestrian_With_Object=2,
        Garbage_Containers_on_Wheels=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        pad_dim=1,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=2,
        load_dim=4,
        pad_dim=1,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        pad_dim=1,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=2,
        load_dim=4,
        pad_dim=1,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=[
         'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='cadc_train_infos_v2.pkl',
            pipeline=train_pipeline,
            metainfo=dict(classes=class_names),
            data_prefix=data_prefix,
            modality=dict(use_lidar=True, use_camera=False),
            with_velocity=True,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

train_cfg = dict(val_interval=1)

default_hooks=dict(
    logger=dict(interval=500), 
    checkpoint=dict(interval=10),
    visualization=dict(
        draw_gt=False,
        draw_pred=True
    )
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                dict(type='WandbVisBackend',
                     init_kwargs=dict(project="centerpoint_cadc"),)
                ]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from='ckpts/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_052355-a6928835.pth'
# load_from = 'work-dir/centerpoint_cadc/2024-07-10/epoch_20.pth'
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(
        pts_voxel_encoder=dict(lr_mult=0.1),
        pts_middle_encoder=dict(lr_mult=0.1),
        pts_backbone=dict(lr_mult=0.1),
        pts_neck=dict(lr_mult=0.1),
        ))
)