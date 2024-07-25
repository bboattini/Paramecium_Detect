from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

BEST = '/home/bernardo/workspace/Paramecium_Detect/app/best.pt'
LAST = '/home/bernardo/workspace/Paramecium_Detect/app/last.pt'
TEST = '/home/bernardo/workspace/Paramecium_Detect/app/Test.mp4'

# Polygon Zone Parameters
HEIGHT = (720, 1200)
WIDTH = (270, 810)
# Polygon mask made in https://roboflow.github.io/polygonzone/
REGION1 = np.array([[164, 170],[164, 422],[544, 422],[544, 170],[164, 170]])
REGION2 = np.array([[WIDTH[0], HEIGHT[0]],[WIDTH[0], HEIGHT[1]],[WIDTH[1], HEIGHT[1]],[WIDTH[1], HEIGHT[0]],[WIDTH[0], HEIGHT[0]]])

def main():
    model = YOLO(f'{LAST}') # call your Trained model for paramecium
    #result = model.track(f'{TEST}', show=True, save_dir='output')

    square_zone = sv.PolygonZone(polygon = REGION2)
    square_zone_annotatator = sv.PolygonZoneAnnotator(
        zone=square_zone,
        thickness=2,
        text_thickness=2,
        text_scale=1,
        color=sv.Color.RED,
    )

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=1,
        text_thickness=2,
    )

    for result in model.track(f'{TEST}', stream=True): # stream=True for video processing
        
        frame = result.orig_img # Get the current image from result generator        
        detections = sv.Detections.from_ultralytics(result) # Convert the result to Detections object

        if result.boxes.id is not None: # If the tracker id is available
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int) # Set the tracker id to the detections object
        
        frame = box_annotator.annotate(scene=frame, detections=detections) # Annotate the image with the bounding boxes
        
        labels = [
            f'{tracker_id} cell {confidence:0.2f}' for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
                  ]

        frame = label_annotator.annotate(scene=frame, detections=detections, labels = labels) # Annotate the image with the labels

        square_zone.trigger(detections=detections) # Trigger the polygon zone with the detections
        square_zone_annotatator.annotate(scene=frame) # Annotate the image with the polygon zone

        cv2.imshow('Paramecium Detect', frame)

        if (cv2.waitKey(30) == 27): # ESC key stops the program execution
            break

        '''print(result)
        print(result.xyxy)
        print(result.tracks)
        print(result.trajectory)'''

if __name__ == '__main__':
    main()