import cv2
import torch
from time import time
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # return 0 if the boxes do not intersect
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # calculate the intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def setup_deepsort(config_path, device):
    """Configure the Deep SORT tracker using settings from YAML file."""
    cfg = get_config()
    cfg.merge_from_file(config_path)
    # initialize Deep SORT with the configurations
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=False
    )
    return deepsort


class Tracker:
    def __init__(self, model_name='yolov5m', device='cpu', config_path='deep_sort/configs/deep_sort.yaml'):
        """Initialize both the detector and the Deep SORT tracker."""
        self.detector = Detector(model_name, device)  # detector initialized with YOLOv5
        self.deepsort = setup_deepsort(config_path, device)  # deep SORT tracker
        self.track_start_times = {}  # dictionary to keep track of when tracking started for each ID
        self.track_trajectories = {}  # dictionary to store trajectories
        self.interactions = []  # list to store interactions between people

    def update(self, frame, frame_idx):
        """Process a single frame for object detection and tracking."""
        _, results = self.detector.detect_objects(frame)  # detection results from YOLOv5
        people = self.detector.filter_people(results)  # filter people from detections

        bbox_xywh = []
        confidences = []

        for person in people:
            x1, y1, x2, y2, conf, _ = person
            # calculate the center points cx, cy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # compute the width and height of the bounding box
            w = x2 - x1
            h = y2 - y1
            bbox_xywh.append([cx, cy, w, h])
            # append the confidence of the detection
            confidences.append(conf)

        if bbox_xywh:
            bbox_xywh = torch.tensor(bbox_xywh)
            confidences = torch.tensor(confidences)
            # performs the tracking operation, updating the track states based on the new detections
            outputs = self.deepsort.update(bbox_xywh, confidences, frame)
            self.detect_interactions(outputs)
            frame = self.vis_track(frame, outputs)

            # prepare analytics data
            analytics_data = {
                'num_people': len(outputs),  # number of people currently being tracked
                'crowd_density': self.calculate_crowd_density(outputs, frame.shape[:2]),  # current crowd density
                'track_durations': self.update_track_durations(outputs, frame_idx)  # duration of each track
            }

        return frame, analytics_data

    def calculate_crowd_density(self, tracks, frame_size):
        """Calculate the crowd density based on the bounding boxes of the tracks."""
        frame_area = frame_size[0] * frame_size[1]
        total_box_area = 0

        for track in tracks:
            # extract bounding box coordinates
            x1, y1, x2, y2 = track[:4]
            # calculate width and height of the box ensuring non-negative values
            width = x2 - x1
            height = y2 - y1
            # update total area
            total_box_area += width * height

        # ensure the box area does not exceed the frame area
        total_box_area = min(total_box_area, frame_area)

        # calculate the density
        density = (total_box_area / frame_area) * 100
        return density

    def update_track_durations(self, tracks, frame_idx):
        """Update the duration of each track and remove ended tracks."""
        if frame_idx is None:
            # skip track duration update if frame_idx is None
            return {}

        """Update the duration of each track and remove ended tracks."""
        current_ids = set(track[4] for track in tracks)
        # initialize start time for new tracks
        for track_id in current_ids:
            if track_id not in self.track_start_times:
                self.track_start_times[track_id] = frame_idx

        # calculate durations for current tracks
        track_durations = {track_id: frame_idx - self.track_start_times[track_id] for track_id in current_ids}

        # remove ended tracks (not present in the current frame)
        ended_tracks = set(self.track_start_times.keys()) - current_ids
        for ended in ended_tracks:
            del self.track_start_times[ended]

        return track_durations

    def detect_interactions(self, tracked_objects):
        """Update to use IoU for interaction detection."""
        self.interactions.clear()
        iou_threshold = 0.1  # threshold for considering interaction

        for i in range(len(tracked_objects)):
            for j in range(i + 1, len(tracked_objects)):
                if calculate_iou(tracked_objects[i], tracked_objects[j]) > iou_threshold:
                    self.interactions.append((tracked_objects[i][4], tracked_objects[j][4]))

    def vis_track(self, img, boxes):
        """visualize the tracking results on the image."""
        normal_color = (255, 0, 0)  # red color in BGR format for non-interacting individuals
        interaction_color = (0, 255, 0)  # green color in BGR format for interacting individuals
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_color = (255, 255, 255)  # white for text

        interacting_ids = [id for interaction in self.interactions for id in interaction]  # flatten list of tuples

        for i in range(len(boxes)):
            box = boxes[i]
            track_id = box[4]

            # check if the current track ID is in any interaction
            if track_id in interacting_ids:
                box_color = interaction_color
            else:
                box_color = normal_color

            # draw bounding box
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # draw the rectangle
            cv2.rectangle(img, (x0, y0), (x1, y1), box_color, 2)
            text = f'{track_id}'
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            # draw a filled rectangle for the text
            cv2.rectangle(img, (x0, y0 - txt_size[1] - 10), (x0 + txt_size[0], y0), box_color, -1)
            cv2.putText(img, text, (x0, y0 - 5), font, 0.4, txt_color, 1)

            # update the trajectory points
            center_point = (int((x0 + x1) / 2), int((y0 + y1) / 2))
            if track_id not in self.track_trajectories:
                self.track_trajectories[track_id] = [center_point]
            else:
                self.track_trajectories[track_id].append(center_point)

            # draw the trajectory line
            for j in range(1, len(self.track_trajectories[track_id])):
                if self.track_trajectories[track_id][j - 1] and self.track_trajectories[track_id][j]:
                    # draw the line
                    cv2.line(img, self.track_trajectories[track_id][j - 1], self.track_trajectories[track_id][j],
                             normal_color, 2)

        return img


# example usage of the Tracker class
if __name__ == '__main__':
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.update(frame, time())
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(0):
            break

    cap.release()
    cv2.destroyAllWindows()
