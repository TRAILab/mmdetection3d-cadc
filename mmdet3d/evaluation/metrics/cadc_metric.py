import json
import os
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
import tqdm
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import _get_box_class_field, load_prediction
from nuscenes.eval.detection.data_classes import (DetectionBox,
                                                  DetectionConfig,
                                                  DetectionMetricDataList,
                                                  DetectionMetrics)
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)
from .nuscenes_metric import output_to_nusc_box


@METRICS.register_module()
class CADCMetric(BaseMetric):
    ALL_CLASSES = [
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
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(
            self,
            data_root: str,
            ann_file: str,
            classes: List[str],
            modality: dict = dict(use_camera=False, use_lidar=True),
            format_only: bool = False,
            jsonfile_prefix: str = 'work-dir/cadc_val_results',
            vel_thresh: float = 0.2,
            backend_args: Optional[dict] = None,
            config_name: str = 'detection_cadc',
            collect_device: str = 'cpu',
            prefix: Optional[str] = None,
            verbose: bool = False,):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root
        self.eval_classes = classes
        self.modality = modality
        if format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'
        self.format_only = format_only
        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args
        self.vel_thresh = vel_thresh
        # load config
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, 'configs', '%s.json' % config_name)
        assert os.path.exists(cfg_path), \
            'Requested unknown configuration {}'.format(config_name)
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        self.eval_det_config = CADCDetectionConfig.deserialize(data)
        self.verbose = verbose

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']
        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        ap_dict = self.cadc_evaluate(
            result_dict, classes=classes, logger=logger)
        for result in ap_dict:
            metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def cadc_evaluate(self,
                      result_dict: dict,
                      classes: List[str],
                      logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in Nuscenes protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated. Defaults to 'bbox'.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        metric_dict = dict()
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(
                result_dict[name], classes=classes, result_name=name, logger=logger)
            metric_dict.update(ret_dict)
        return metric_dict

    def _evaluate_single(
            self,
            result_path: str,
            classes: List[str],
            result_name: str = 'pred_instances_3d',
            logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])

        cadc_eval = CADCDetectionEval(
            config=self.eval_det_config,
            gt_infos=self.data_infos,
            result_path=result_path,
            output_dir=output_dir,
            verbose=self.verbose,
            logger=logger)
        cadc_eval.main(render_curves=False)

        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(
        self,
        results: List[dict],
        classes: List[str],
        jsonfile_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                box_type_3d = type(results_[0]['bboxes_3d'])
                if box_type_3d == LiDARInstance3DBoxes:
                    result_dict[name] = self._format_lidar_bbox(
                        results_, sample_idx_list, classes, tmp_file_)
                elif box_type_3d == CameraInstance3DBoxes:
                    raise NotImplementedError
                    # result_dict[name] = self._format_camera_bbox(
                    # results_, sample_idx_list, classes, tmp_file_)

        return result_dict, tmp_dir

    def _format_lidar_bbox(self,
                           results: List[dict],
                           sample_idx_list: List[int],
                           classes: List[str],
                           jsonfile_prefix: str) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']
            boxes = lidar_box_to_global(self.data_infos[sample_idx], boxes)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    ego_translation=det['bboxes_3d'][i].gravity_center.numpy().reshape(
                        3).tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name='',)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results writes to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path


class CADCDetectionEval(NuScenesEval):
    def __init__(self,
                 config: DetectionConfig,
                 gt_infos,
                 result_path: str,
                 output_dir: str,
                 verbose: bool = True,
                 logger: Optional[MMLogger] = None):
        """
        Initialize a DetectionEval object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.result_path = result_path
        self.output_dir = output_dir
        self.verbose = verbose
        if verbose:
            assert logger is not None, 'logger must be specified when verbose is True'
        self.logger = logger
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(
            result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            self.logger.info('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            CADCDetectionBox,
            verbose=verbose)
        self.gt_boxes = load_gt(
            gt_infos,
            CADCDetectionBox,
            verbose=verbose,
            logger=logger)
        # sanity check by setting pred boxes to gt boxes
        # self.pred_boxes = self.gt_boxes

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            self.logger.info('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(
            self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            self.logger.info('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(
            self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens


class CADCDetectionConfig(DetectionConfig):
    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):
        # do not require class_range keys to match nuscenes detection names
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()


class CADCDetectionBox(DetectionBox):
    def __init__(
            self,
            sample_token: str = "",
            translation: Tuple[float, float, float] = (0, 0, 0),
            size: Tuple[float, float, float] = (0, 0, 0),
            rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
            velocity: Tuple[float, float] = (0, 0),
            # Translation to ego vehicle in meters.
            ego_translation: Tuple[float, float, float] = (0, 0, 0),
            # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
            num_pts: int = -1,
            # The class name used in the detection challenge.
            detection_name: str = 'car',
            detection_score: float = -1.0,  # GT samples do not have a score.
            attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size,
                         rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        # no longer check if detection_name is in nuscenes detection names

        assert type(
            detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)
                          ), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name


def lidar_box_to_global(
        info: dict, boxes: List[NuScenesBox]) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # TODO filter det in ego based on distance
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list


def load_gt(gt_infos, box_cls, verbose: bool = False, logger: Optional[MMLogger] = None) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if verbose:
        logger.info('Loading annotations from CADC')
    # Read out all sample_tokens in DB.
    assert len(gt_infos) > 0, "Error: Database has no samples!"

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for gt_info in tqdm.tqdm(gt_infos, leave=verbose):
        sample_boxes = []
        lidar2ego = np.array(gt_info['lidar_points']['lidar2ego'])
        lidar2ego_quat = pyquaternion.Quaternion(
            matrix=lidar2ego, rtol=1e-05, atol=1e-07)
        ego2global = np.array(gt_info['ego2global'])
        ego2global_quat = pyquaternion.Quaternion(
            matrix=ego2global, rtol=1e-05, atol=1e-07)
        for instance in gt_info['instances']:
            # convert gt info to nuScenes box format
            # nuscenes box takes size as (w, l, h) while cadc box is size as (l, w, h)
            nus_box = NuScenesBox(
                center=instance['bbox_3d'][0:3],
                size=[instance['bbox_3d'][i] for i in [4, 3, 5]],
                orientation=pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=instance['bbox_3d'][6]),
                velocity=(instance['velocity'][0], instance['velocity'][1], 0)
            )
            # convert gt info from lidar frame to global frame
            nus_box.rotate(lidar2ego_quat)
            nus_box.translate(lidar2ego[:3, 3])
            nus_box.rotate(ego2global_quat)
            nus_box.translate(ego2global[:3, 3])
            if issubclass(box_cls, DetectionBox):
                # Get label name in detection task and filter unused labels.
                detection_name = instance['bbox_label_3d']
                if detection_name is None:
                    continue
                sample_boxes.append(
                    box_cls(
                        sample_token=gt_info['token'],
                        translation=nus_box.center.tolist(),
                        size=nus_box.wlh.tolist(),
                        rotation=nus_box.orientation.elements.tolist(),
                        velocity=nus_box.velocity[:2],
                        ego_translation=instance['bbox_3d'][0:3],
                        num_pts=instance['num_lidar_pts'],
                        detection_name=detection_name,
                        # GT samples do not have a score.
                        detection_score=-1.0,
                        # detection_score=1.0,
                    )
                )
            elif box_cls == TrackingBox:
                raise NotImplementedError(
                    'Error: TrackingBox not supported in CADC!')
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import \
                    category_to_tracking_name
                tracking_name = category_to_tracking_name(
                    sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(
                            sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] +
                        sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError(
                    'Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(gt_info['token'], sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(
            len(all_annotations.sample_tokens)))

    return all_annotations


def filter_eval_boxes(
        eval_boxes: EvalBoxes,
        max_dist: Dict[str, float],
        verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter = 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

    return eval_boxes
