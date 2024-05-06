import torch
import cv2


class Detector:
    def __init__(self, model_name='yolov5m', device= None):
        """Initialize the detector with a specified YOLOv5 model and device."""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.eval().to(self.device)

    def detect_objects(self, frame):
        """Detect objects in an image"""
        if frame is None:
            raise ValueError(f"Image at path {frame} could not be read.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = self.model([frame_rgb])

        return frame, results

    def filter_people(self, results):
        # Assuming '0' is the class for 'person'
        people = [x for x in results.xyxy[0] if int(x[5]) == 0]
        return people

    def show_results(self, frame, results):
        """Visualize the detection results on the frame."""
        # check if detections were made
        if len(results.xyxy[0]) > 0:
            # draw bounding boxes from detections
            for det in results.xyxy[0]:  # detections for the first image in batch
                x1, y1, x2, y2, conf, cls = det
                if cls == 0:  # '0' is the class for 'person'
                    # draw rectangles on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        cv2.imwrite('detect1.jpg', frame)


# example usage of the Detector class
if __name__ == '__main__':
    detector = Detector(model_name='yolov5m', device='cpu')
    image_path = 'imgs_test/000005.jpg'
    frame = cv2.imread(image_path)
    frame, results = detector.detect_objects(frame)
    detector.show_results(frame, results)