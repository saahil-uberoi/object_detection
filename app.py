# Building REST APIs using Python and Flask

import time
import cv2
import os
import numpy as np

from flask import Flask, jsonify, request

app = Flask(__name__)

CONFIDENCE_PARAMETER = 0.5
YOLO_BASE_DIR = os.getcwd() + '/yolo-coco'
LABELS = open(os.path.sep.join([YOLO_BASE_DIR, "coco.names"])).read().strip().split("\n")


@app.route('/API/flask_server', methods=['POST'])
def process():
    image_obj = request.files.get('image')
    npimg = np.fromstring(image_obj.read(), np.uint8)
    image = imdecode(npimg, IMREAD_UNCHANGED)
    weights_path = os.path.sep.join([YOLO_BASE_DIR, "yolov3.weights"])
    config_path = os.path.sep.join([YOLO_BASE_DIR, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_PARAMETER:
                confidences.append(float(confidence))
                class_ids.append(class_id)

    response = []
    for i, j in zip(confidences, class_ids):
        result = {'label': LABELS[j],
                  'accuracy': i * 100
                  }
        response.append(result)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')