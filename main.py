# main.py
import sys
sys.path.append(r'F:\file\University\object\Mask\models')
sys.path.append(r'F:\file\University\object\Mask\models\research')
sys.path.append(r'F:\file\University\object\Mask\models\research\slim')



import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# ------------------ TFOD API PATH ------------------


from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

print("TFOD API ready!")

# ------------------ Paths ------------------
image_dir = 'Data/images'
annotations_dir = 'Data/annotations'
train_record_path = 'Data/train.record'
label_map_path = 'Data/label_map.pbtxt'
pipeline_config_path = 'Data/ssd_mobilenet_v2.config'  # bạn chuẩn bị config từ TFOD Model Zoo
model_dir = 'Data/model'

# ------------------ Labels ------------------
label_map = {'with_mask': 1, 'without_mask': 2, 'mask_weared_incorrect': 3}

# ------------------ Create label_map.pbtxt ------------------
with open(label_map_path, 'w') as f:
    for name, idx in label_map.items():
        f.write("item {\n")
        f.write(f"  id: {idx}\n")
        f.write(f"  name: '{name}'\n")
        f.write("}\n")
print(f"Label map created at {label_map_path}")

# ------------------ Create TFExample ------------------
def create_tf_example(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(path, 'rb') as fid:
        encoded_image = fid.read()
    
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map[class_name])
        
        bndbox = obj.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'png'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# ------------------ Generate TFRecord ------------------
writer = tf.io.TFRecordWriter(train_record_path)
for xml_file in glob.glob(os.path.join(annotations_dir, '*.xml')):
    tf_example = create_tf_example(xml_file)
    writer.write(tf_example.SerializeToString())
writer.close()
print('TFRecord created successfully!')

# ------------------ Build SSD model template ------------------
if not os.path.exists(pipeline_config_path):
    print("Pipeline config file not found. Please download SSD config from TFOD Model Zoo and place it in Data/")
else:
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    print("SSD model built. Ready for training template!")

# ------------------ Inference test template ------------------
# Chạy thử trên 1 ảnh đầu tiên
test_images = glob.glob(os.path.join(image_dir, '*.png'))
if len(test_images) > 0:
    image_path = test_images[0]
    image_np = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # Nếu bạn đã train và export saved_model, thay đổi đường dẫn ở đây
    saved_model_dir = os.path.join(model_dir, 'saved_model')
    if os.path.exists(saved_model_dir):
        detect_fn = tf.saved_model.load(saved_model_dir)
        detections = detect_fn(input_tensor)
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(int),
            detections['detection_scores'][0].numpy(),
            label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True),
            use_normalized_coordinates=True,
            line_thickness=2
        )
        cv2.imshow('Detection Test', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No trained model found. Skipping inference test.")
else:
    print("No images found for inference test.")
