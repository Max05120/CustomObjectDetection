import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200713'
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)




# IMAGE_PATH = os.path.join('/home/max/tensorflow/dataset/label8.jpeg')
category_index = label_map_util.create_category_index_from_labelmap(os.path.join('/home/max/tensorflow/dataset/lakers_label_map.pbtxt'))
configs = config_util.get_configs_from_pipeline_file('/home/max/tensorflow/training_demo/models/my_ssd_resnet50_fpn/pipeline.config')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)


# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join('/home/max/tensorflow/training_demo/models/my_ssd_resnet50_fpn','ckpt-6')).expect_partial()
PATH_TO_SAVED_MODEL = '/home/max/tensorflow/training_demo/exported_models/my_model/saved_model'

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

# def get_model_detection_function(model):


#     @tf.function
#     def detect_fn(image):
#         """Detect objects in image."""

#         image, shapes = model.preprocess(image)
#         prediction_dict = model.predict(image, shapes)
#         detections = model.postprocess(prediction_dict, shapes)

#         return detections, prediction_dict, tf.reshape(shapes, [-1])

#     return detect_fn

# detect_fn = get_model_detection_function(detection_model)

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



while cap.isOpened():
    ret,frame = cap.read()
    image_np =np.array(frame)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = input_tensor[:, :, :]
    detections =detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


    label_id_offset = 1
    image_np_with_detections = image_np.copy()


    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0,
        agnostic_mode=False
    )

    cv2.imshow('object_detection',cv2.resize(image_np_with_detections,(800,600)))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    

