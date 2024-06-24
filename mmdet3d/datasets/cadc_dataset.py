from typing import Callable, List, Optional, Union

import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes, get_box_type
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class CADCDataset(Det3DDataset):

    # replace with all the classes in customized pkl info file
    METAINFO = {
        'classes': (
            'Car', 
            'Pedestrian', 
            'Truck', 
            'Bus', 
            'Garbage_Containers_on_Wheels', 
            'Traffic_Guidance_Objects',
            'Bicycle',
            'Pedestrian_With_Object', 
            'Horse_and_Buggy',
            'Animals'
            ),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        self.backend_args = backend_args
        self.filter_empty_gt = filter_empty_gt
        self.load_eval_anns = load_eval_anns
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()

        # Defaults to False if not specify
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        self.default_cam_key = default_cam_key
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f', `use_camera`) for {self.__class__.__name__}')

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        if metainfo is not None and 'classes' in metainfo:
            # we allow to train on subset of self.METAINFO['classes']
            # map unselected labels to -1
            self.label_mapping = {
                name: -1
                for i, name in enumerate(self.METAINFO['classes'])
            }
            self.label_mapping[-1] = -1
            curr_label = 0
            for label_idx, name in enumerate(metainfo['classes']):
                assert name in self.METAINFO['classes'], f'Class name {name} not in possible classes: {self.METAINFO["classes"]}'
                self.label_mapping[name] = curr_label
                curr_label += 1

            self.num_ins_per_cat = [0] * len(metainfo['classes'])
        else:
            self.label_mapping = {
                name: i
                for i in enumerate(self.METAINFO['classes'])
            }
            self.label_mapping[-1] = -1

            self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])

        super(Det3DDataset, self).__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

        # can be accessed by other component in runner
        self.metainfo['box_type_3d'] = box_type_3d
        self.metainfo['label_mapping'] = self.label_mapping

        if not kwargs.get('lazy_init', False):
            # used for showing variation of the number of instances before and
            # after through the pipeline
            self.show_ins_var = show_ins_var

            # show statistics of this dataset
            print_log('-' * 30, 'current')
            print_log(
                f'The length of {"test" if self.test_mode else "training"} dataset: {len(self)}',  # noqa: E501
                'current')
            content_show = [['category', 'number']]
            for label, num in enumerate(self.num_ins_per_cat):
                cat_name = self.metainfo['classes'][label]
                content_show.append([cat_name, num])
            table = AsciiTable(content_show)
            print_log(
                f'The number of instances per category in the dataset:\n{table.table}',  # noqa: E501
                'current')

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:/home/brian/Documents/Projects/PF-Track2/projects/mmengine_plugin/datasets/cadc_dataset.py
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
