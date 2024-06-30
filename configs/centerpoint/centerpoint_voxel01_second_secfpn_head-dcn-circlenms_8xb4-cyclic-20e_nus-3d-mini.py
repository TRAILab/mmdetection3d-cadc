_base_ = ['./centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py']


data_root = 'data/nuscenes/'
train_dataloader = dict(
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
    )
)
test_dataloader = dict(
    dataset=dict(
        ann_file='nuscenes_track_infos_val.pkl',
    )
)
val_dataloader = dict(
    dataset=dict(
        ann_file='nuscenes_track_infos_val.pkl',
    )
)

val_evaluator = dict(
    ann_file=data_root + 'nuscenes_track_infos_val.pkl',
)
test_evaluator = dict(
    ann_file=data_root + 'nuscenes_track_infos_val.pkl',
)
