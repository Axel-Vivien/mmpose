# Copyright (c) OpenMMLab. All rights reserved.
"""MMPose Inferencer Demo - Process videos from local files or S3 with on-demand download/cleanup."""

from argparse import ArgumentParser
from typing import Dict
import json
import cv2
import os
import glob
import time
import threading
import statistics
import multiprocessing as mp
import tempfile

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

PYNVML_AVAILABLE = False
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']


class PerformanceMonitor:
    def __init__(self, monitor_interval=0.1):
        self.monitor_interval = monitor_interval
        self.cpu_usage = []
        self.gpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                self.gpu_available = False
        elif GPUTIL_AVAILABLE:
            try:
                GPUtil.getGPUs()
                self.gpu_available = True
            except:
                self.gpu_available = False
        else:
            self.gpu_available = False
    
    def start_monitoring(self):
        self.cpu_usage = []
        self.gpu_usage = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        stats = {}
        if PSUTIL_AVAILABLE and self.cpu_usage:
            stats.update({
                'cpu_avg': statistics.mean(self.cpu_usage),
                'cpu_max': max(self.cpu_usage),
                'cpu_min': min(self.cpu_usage)
            })
        else:
            stats.update({'cpu_avg': None, 'cpu_max': None, 'cpu_min': None})
        
        if self.gpu_available and self.gpu_usage:
            stats.update({
                'gpu_avg': statistics.mean(self.gpu_usage),
                'gpu_max': max(self.gpu_usage),
                'gpu_min': min(self.gpu_usage)
            })
        else:
            stats.update({'gpu_avg': None, 'gpu_max': None, 'gpu_min': None})
        
        return stats
    
    def _monitor_loop(self):
        while self.monitoring:
            if PSUTIL_AVAILABLE:
                try:
                    self.cpu_usage.append(psutil.cpu_percent(interval=None))
                except:
                    pass
            
            if self.gpu_available:
                try:
                    if GPUTIL_AVAILABLE:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            self.gpu_usage.append(gpus[0].load * 100)
                    elif PYNVML_AVAILABLE:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_usage.append(util.gpu)
                except:
                    pass
            
            time.sleep(self.monitor_interval)


def calculate_fps(frame_count, processing_time):
    return frame_count / processing_time if processing_time > 0 else 0


def process_video_worker(worker_info):
    video_item, init_args, call_args_template, process_id = worker_info
    
    try:
        is_on_demand = isinstance(video_item, tuple)
        inferencer = MMPoseInferencer(**init_args)
        
        if is_on_demand:
            perf_data = process_single_video_on_demand(video_item, inferencer, call_args_template)
        else:
            perf_data = process_single_video(video_item, inferencer, call_args_template, None)
        
        perf_data['process_id'] = process_id
        return perf_data
        
    except Exception as e:
        error_path = video_item[0] if is_on_demand else video_item
        return {
            'fps': 0,
            'stats': {'cpu_avg': None, 'cpu_max': None, 'cpu_min': None, 
                     'gpu_avg': None, 'gpu_max': None, 'gpu_min': None},
            'processing_time': 0, 'frame_count': 0, 'error': str(e),
            'process_id': process_id
        }


def process_videos_parallel(video_tasks, init_args, call_args, num_processes):
    work_items = []
    for i, video_item in enumerate(video_tasks):
        process_id = (i % num_processes) + 1
        work_items.append((video_item, init_args, call_args, process_id))
    
    with mp.Pool(processes=num_processes) as pool:
        try:
            return pool.map(process_video_worker, work_items)
        except (KeyboardInterrupt, Exception):
            pool.terminate()
            pool.join()
            return []


def process_videos_sequential(video_tasks, init_args, call_args):
    inferencer = MMPoseInferencer(**init_args)
    all_performance_data = []
    is_on_demand = len(video_tasks) > 0 and isinstance(video_tasks[0], tuple)
    
    for video_item in video_tasks:
        if is_on_demand:
            perf_data = process_single_video_on_demand(video_item, inferencer, call_args)
        else:
            perf_data = process_single_video(video_item, inferencer, call_args, None)
        
        all_performance_data.append(perf_data)
    
    return all_performance_data


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or folder path. Optional when using --json-file.')

    # init args
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Display the progress bar during inference.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # call args
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=4,
        help='Number of processes to use for parallel video processing.')

    # JSON processing arguments
    parser.add_argument(
        '--json-file',
        type=str,
        default=None,
        help='Path to action_timespans.json file for video processing.')
    parser.add_argument(
        '--category',
        type=str,
        choices=['true_positive', 'false_positive', 'unbiased', 'all'],
        default='all',
        help='Video category to process from JSON file.')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of videos to process from JSON file.')
    parser.add_argument(
        '--download-dir',
        type=str,
        default='./downloaded_videos',
        help='Directory to download videos from S3.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    display_alias = call_args.pop('show_alias')
    num_processes = call_args.pop('num_processes')
    
    # JSON processing arguments
    json_file = call_args.pop('json_file')
    category = call_args.pop('category')
    limit = call_args.pop('limit')
    download_dir = call_args.pop('download_dir')
    
    # hard-coded values
    init_args['pose2d'] = "rtmw-x_8xb320-270e_cocktail14-384x288"
 
    return init_args, call_args, display_alias, num_processes, json_file, category, limit, download_dir


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')


def get_video_files(input_path):
    """Get list of video files from input path.
    
    Args:
        input_path (str): Path to video file or directory containing videos
        
    Returns:
        list: List of video file paths
    """
    if os.path.isfile(input_path):
        # Single file - check if it's a video
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return [input_path]
        else:
            print(f"Warning: {input_path} is not a recognized video file")
            return []
    elif os.path.isdir(input_path):
        # Directory - find all video files
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            pattern = os.path.join(input_path, f"*{ext}")
            video_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(input_path, f"*{ext.upper()}")
            video_files.extend(glob.glob(pattern))
        
        video_files.sort()  # Sort for consistent processing order
        if not video_files:
            print(f"No video files found in directory: {input_path}")
        else:
            print(f"Found {len(video_files)} video files in {input_path}")
        return video_files
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []


def process_single_video_on_demand(video_task, inferencer, call_args_template):
    s3_path, local_temp_path, video_metadata = video_task
    video_filename = os.path.basename(local_temp_path)
    
    download_start = time.time()
    if not download_video_from_s3(s3_path, local_temp_path):
        return {
            'video_path': s3_path, 'local_path': local_temp_path, 'fps': 0,
            'stats': {'cpu_avg': None, 'cpu_max': None, 'cpu_min': None, 
                     'gpu_avg': None, 'gpu_max': None, 'gpu_min': None},
            'processing_time': 0, 'frame_count': 0, 'download_time': 0,
            'error': 'Download failed'
        }
    
    download_time = time.time() - download_start
    
    temp_dir = os.path.dirname(local_temp_path)
    results_dir = os.path.join(temp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    call_args_with_results = call_args_template.copy()
    call_args_with_results['pred_out_dir'] = results_dir
    
    # Prepare metadata for inclusion in result file
    metadata = {
        's3_path': s3_path,
        'original_metadata': video_metadata
    }
    
    processing_result = process_single_video(local_temp_path, inferencer, call_args_with_results, metadata)
    
    video_basename = os.path.splitext(video_filename)[0]
    json_filename = video_basename + '.json'
    json_path = os.path.join(results_dir, json_filename)
    
    # Initialize upload tracking
    upload_success = False
    s3_results_key = f"axel/ffmpose_results/data-sample_2025-06-02/{json_filename}"
    
    # Add metadata to the result JSON file and upload to S3
    if os.path.exists(json_path):
        # Get video properties that were captured during processing
        cap = cv2.VideoCapture(local_temp_path)
        video_properties = None
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            video_properties = {
                "frame_count": frame_count, "frame_width": frame_width,
                "frame_height": frame_height, "frame_rate": frame_rate,
            }
        
        # Add metadata directly to the results JSON file
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            video_info = {}
            
            # Add video properties
            if video_properties:
                video_info["video_properties"] = video_properties

            # Add source metadata
            s3_path = metadata.get('s3_path')
            original_metadata = metadata.get('original_metadata', {})
            
            category = 'unknown'
            if s3_path:
                if 'true_positive' in s3_path:
                    category = 'true_positive'
                elif 'false_positive' in s3_path:
                    category = 'false_positive'
                elif 'unbiased' in s3_path:
                    category = 'unbiased'
            
            video_info["source_metadata"] = {
                "s3_path": s3_path,
                "category": category,
                "original_json_metadata": original_metadata
            }

            # Insert metadata at the beginning and upload to S3
            if video_info and isinstance(data, list):
                data.insert(0, video_info)
                
                # Upload directly to S3 instead of saving locally
                upload_success = upload_results_to_s3(data, s3_results_key)
                    
        except:
            pass
    
    # Cleanup: remove temporary video and local results
    try:
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        # Try to remove results directory if empty
        try:
            os.rmdir(results_dir)
        except:
            pass
    except:
        pass
    
    processing_result.update({
        's3_path': s3_path,
        'video_metadata': video_metadata,
        'download_time': download_time,
        'local_temp_path': local_temp_path,
        'upload_success': upload_success,
        's3_results_key': s3_results_key if upload_success else None
    })
    
    return processing_result


def process_single_video(video_path, inferencer, call_args_template, metadata=None):
    # Capture video properties while video exists
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_properties = None
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        video_properties = {
            "frame_count": frame_count, "frame_width": frame_width,
            "frame_height": frame_height, "frame_rate": frame_rate,
        }
    
    call_args = call_args_template.copy()
    call_args['inputs'] = video_path
    call_args['pred_out_dir'] = call_args_template.get('pred_out_dir', os.path.dirname(video_path))
    
    monitor = PerformanceMonitor()
    
    try:
        monitor.start_monitoring()
        start_time = time.time()
        
        for _ in inferencer(**call_args):
            pass
        
        end_time = time.time()
        processing_time = end_time - start_time
        stats = monitor.stop_monitoring()
        fps = calculate_fps(frame_count, processing_time)
        
        # Pass video properties to ensure they're added even if video is deleted
        add_video_properties_to_json(video_path, metadata, video_properties)
        
        return {
            'fps': fps, 'stats': stats,
            'processing_time': processing_time, 'frame_count': frame_count
        }
        
    except Exception as e:
        try:
            stats = monitor.stop_monitoring()
        except:
            stats = {'cpu_avg': None, 'cpu_max': None, 'cpu_min': None, 
                    'gpu_avg': None, 'gpu_max': None, 'gpu_min': None}
        
        return {
            'fps': 0, 'stats': stats,
            'processing_time': 0, 'frame_count': frame_count, 'error': str(e)
        }


def add_video_properties_to_json(video_path, metadata=None, video_properties=None):
    pre, ext = os.path.splitext(video_path)
    json_filepath = pre + '.json'
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        video_info = {}
        
        # Use provided video properties or try to get them from video file
        if video_properties:
            video_info["video_properties"] = video_properties
        else:
            # Try to get video properties, but don't fail if video can't be opened
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                video_info["video_properties"] = {
                    "frame_count": frame_count, "frame_width": frame_width,
                    "frame_height": frame_height, "frame_rate": frame_rate,
                }

        # Add metadata from original JSON if available
        if metadata:
            s3_path = metadata.get('s3_path')
            original_metadata = metadata.get('original_metadata', {})
            
            # Extract category from s3_path
            category = 'unknown'
            if s3_path:
                if 'true_positive' in s3_path:
                    category = 'true_positive'
                elif 'false_positive' in s3_path:
                    category = 'false_positive'
                elif 'unbiased' in s3_path:
                    category = 'unbiased'
            
            video_info["source_metadata"] = {
                "s3_path": s3_path,
                "category": category,
                "original_json_metadata": original_metadata
            }

        # Add video_info to data if we have any information to add
        if video_info:
            if isinstance(data, list):
                data.insert(0, video_info)
            elif isinstance(data, dict):
                if "video_properties" in video_info:
                    data["video_properties"] = video_info["video_properties"]
                if "source_metadata" in video_info:
                    data["source_metadata"] = video_info["source_metadata"]

            with open(json_filepath, 'w') as f:
                json.dump(data, f, indent=4)

    except:
        pass


def main():
    init_args, call_args, display_alias, num_processes, json_file, category, limit, download_dir = parse_args()
    if display_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
        return

    if json_file:
        video_tasks = prepare_videos_from_json(json_file, category, limit, download_dir)
    else:
        if not call_args['inputs']:
            return
        video_tasks = get_video_files(call_args['inputs'])
    
    if not video_tasks:
        return
    
    if num_processes < 1:
        return
    
    max_processes = min(len(video_tasks), mp.cpu_count())
    if num_processes > max_processes:
        num_processes = max_processes
    
    if num_processes > 1:
        all_performance_data = process_videos_parallel(video_tasks, init_args, call_args, num_processes)
    else:
        all_performance_data = process_videos_sequential(video_tasks, init_args, call_args)
    
    all_performance_data = [data for data in all_performance_data if data is not None]
    
    # Determine output location for performance summary
    if json_file:
        # For JSON mode, save to current directory or a fixed location
        summary_output_path = '.'
    else:
        # For file/directory mode, use the original logic
        summary_output_path = call_args.get('inputs', '.')
    
    save_performance_summary(all_performance_data, summary_output_path, json_file)


def save_performance_summary(performance_data, input_path, json_file=None):
    try:
        output_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path) if input_path else '.'
        
        # Generate filename based on JSON file name if provided
        if json_file:
            # Extract base name from JSON file path (handle both local and S3 paths)
            if json_file.startswith('s3://'):
                json_basename = os.path.basename(json_file)
            else:
                json_basename = os.path.basename(json_file)
            
            # Remove .json extension and create performance summary filename
            json_name_without_ext = os.path.splitext(json_basename)[0]
            performance_filename = f"performance_summary_{json_name_without_ext}.json"
        else:
            performance_filename = 'performance_summary.json'
        
        output_file = os.path.join(output_dir, performance_filename)
        
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_videos': len(performance_data),
            'videos': [],
            'source_json_file': json_file if json_file else None
        }
        
        for data in performance_data:
            video_summary = {
                'video_name': os.path.basename(str(data.get('s3_path', data.get('video_path', 'unknown')))),
                'frame_count': data['frame_count'],
                'processing_time': data['processing_time'],
                'fps': data['fps'],
                'cpu_avg': data['stats']['cpu_avg'],
                'cpu_max': data['stats']['cpu_max'],
                'cpu_min': data['stats']['cpu_min'],
                'gpu_avg': data['stats']['gpu_avg'],
                'gpu_max': data['stats']['gpu_max'],
                'gpu_min': data['stats']['gpu_min']
            }
            if 'error' in data:
                video_summary['error'] = data['error']
            if 'process_id' in data:
                video_summary['process_id'] = data['process_id']
            if 'upload_success' in data:
                video_summary['upload_success'] = data['upload_success']
            if 's3_results_key' in data:
                video_summary['s3_results_key'] = data['s3_results_key']
            if 's3_path' in data:
                video_summary['s3_path'] = data['s3_path']
            if 'download_time' in data:
                video_summary['download_time'] = data['download_time']
            
            summary_data['videos'].append(video_summary)
        
        fps_values = [data['fps'] for data in performance_data if data['fps'] > 0]
        cpu_avg_values = [data['stats']['cpu_avg'] for data in performance_data if data['stats']['cpu_avg'] is not None]
        gpu_avg_values = [data['stats']['gpu_avg'] for data in performance_data if data['stats']['gpu_avg'] is not None]
        
        summary_data['overall_averages'] = {
            'fps_avg': statistics.mean(fps_values) if fps_values else None,
            'fps_max': max(fps_values) if fps_values else None,
            'fps_min': min(fps_values) if fps_values else None,
            'cpu_avg': statistics.mean(cpu_avg_values) if cpu_avg_values else None,
            'gpu_avg': statistics.mean(gpu_avg_values) if gpu_avg_values else None
        }
        
        # Add S3 upload statistics if available
        upload_successes = [data.get('upload_success', False) for data in performance_data if 'upload_success' in data]
        if upload_successes:
            summary_data['s3_upload_stats'] = {
                'total_uploads_attempted': len(upload_successes),
                'successful_uploads': sum(upload_successes),
                'failed_uploads': len(upload_successes) - sum(upload_successes),
                'success_rate': (sum(upload_successes) / len(upload_successes) * 100) if upload_successes else 0
            }
        
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        # Upload performance summary to S3 if json_file was used
        if json_file:
            s3_key = f"axel/performance_summaries/{performance_filename}"
            try:
                upload_success = upload_results_to_s3(summary_data, s3_key)
                if upload_success:
                    print(f"✅ Performance summary uploaded to S3: s3://veesion-data-reinit-research/{s3_key}")
                else:
                    print(f"❌ Failed to upload performance summary to S3")
            except:
                print(f"❌ Error uploading performance summary to S3")
        
    except:
        pass


def download_json_from_s3(s3_path, local_path, bucket_name="veesion-data-reinit-research"):
    """Download a JSON file from S3"""
    s3_client = get_s3_client()
    if not s3_client:
        return False
        
    try:
        # Extract key from s3://bucket/key format
        if s3_path.startswith('s3://'):
            # Remove s3:// and split bucket/key
            path_parts = s3_path[5:].split('/', 1)
            if len(path_parts) == 2:
                bucket_name = path_parts[0]
                s3_key = path_parts[1]
            else:
                return False
        else:
            s3_key = s3_path
            
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except:
        return False


def load_json_videos(json_file_path):
    """Load JSON videos from local file or S3"""
    # Check if it's an S3 path
    if json_file_path.startswith('s3://'):
        # Download from S3 to temporary location
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        temp_file.close()
        
        if download_json_from_s3(json_file_path, temp_file.name):
            try:
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Clean up temporary file
                os.unlink(temp_file.name)
                return data
            except:
                # Clean up temporary file on error
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                return None
        else:
            # Clean up temporary file if download failed
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return None
    else:
        # Load from local file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None


def filter_videos_by_category(video_data, category):
    if category == 'all':
        return list(video_data.keys())
    return [path for path in video_data.keys() if category in path]


def get_s3_client():
    """Get S3 client with research profile"""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
        
        session = boto3.Session(region_name='eu-west-1')
        return session.client('s3')
    except:
        return None


def download_video_from_s3(s3_path, local_path, bucket_name="veesion-data-reinit-research"):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    s3_client = get_s3_client()
    if not s3_client:
        return False
        
    try:
        s3_client.download_file(bucket_name, s3_path, local_path)
        return True
    except:
        return False


def upload_results_to_s3(json_data, s3_key, bucket_name="veesion-data-reinit-research"):
    """Upload JSON results directly to S3"""
    s3_client = get_s3_client()
    if not s3_client:
        return False
        
    try:
        json_string = json.dumps(json_data, indent=4)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_string.encode('utf-8'),
            ContentType='application/json'
        )
        return True
    except:
        return False





def prepare_videos_from_json(json_file, category, limit, download_dir):
    video_data = load_json_videos(json_file)
    if not video_data:
        return []
    
    video_paths = filter_videos_by_category(video_data, category)
    
    if limit and limit < len(video_paths):
        video_paths = video_paths[:limit]
    
    if not video_paths:
        return []
    
    os.makedirs(download_dir, exist_ok=True)
    
    video_tasks = []
    for s3_path in video_paths:
        video_filename = os.path.basename(s3_path)
        local_temp_path = os.path.join(download_dir, video_filename)
        video_metadata = video_data[s3_path]
        video_tasks.append((s3_path, local_temp_path, video_metadata))
    
    return video_tasks


if __name__ == '__main__':
    # Multiprocessing guard for Windows and other platforms
    mp.set_start_method('spawn', force=True) if hasattr(mp, 'set_start_method') else None
    main()
