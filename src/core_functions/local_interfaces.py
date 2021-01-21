from imutils.video import VideoStream
import cv2
import model_serving
import time
import imutils
import argparse
def mask_image(model_object, image):

    (locs, preds) = model_object.predict_frame(image)

    print(locs,preds)
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.imshow('Output_image',image)
        cv2.waitKey(0)

def mask_video(model_object, video=0):
    vs = VideoStream(video).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = model_object.predict_frame(frame)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i','--image',help="path to input image")
    group.add_argument('-v','--video',help="path to input video")
    parser.add_argument("-f", "--face", type=str,
                    help="path to face detector model directory")
    parser.add_argument("-m", "--mask_model", type=str,
                    help="path to trained face mask detector model")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(parser.parse_args())
    model_object=model_serving.DetectFaceClassifyMask(args['face'], args['mask_model'], args['confidence'])
    model_object.load()
    if (args['image']!=None):
        image= cv2.imread(args['image'])
        mask_image(model_object,image)
    else:
        mask_video(model_object,args['video'])
