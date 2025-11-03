import sys
import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util



CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
image_dir = os.path.join(CURRENT_DIR, 'Data', 'images_test')
annotations_dir = os.path.join(CURRENT_DIR, 'Data', 'annotations_test')
test_record_path = os.path.join(CURRENT_DIR, 'Data', 'test.record')
label_map_path = os.path.join(CURRENT_DIR, 'Data', 'label_map.pbtxt')



label_map = {}
if not os.path.exists(label_map_path):
    print(f"LỖI: không tìm thấy tệp {label_map_path}. Hãy chạy main.py trước!")
    sys.exit()
    
with open(label_map_path, 'r') as f:
    for line in f:
        if "id:" in line:
            label_id = int(line.strip().split(":")[-1])
        if "name:" in line:
            label_name = line.strip().split(":")[-1].strip().replace("'", "")
            label_map[label_name] = label_id
print(f"Đã tải label map: {label_map}")


def create_tf_example(xml_file):
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        print(f"Lỗi: Không thể phân tích tệp XML. Bỏ qua: {xml_file}. Lỗi: {e}")
        return None
        
    root = tree.getroot()
    
    filename = root.find('filename').text
    image_path = os.path.join(image_dir, filename)
    
    
    if not os.path.exists(image_path):
        print(f"Cảnh báo: Tệp ảnh {image_path} được liệt kê trong XML nhưng không tìm thấy. Bỏ qua.")
        return None

   
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
   
    image_format_str = filename.split('.')[-1].lower()
    if image_format_str in ['jpg', 'jpeg']:
        image_format = b'jpeg'
    elif image_format_str == 'png':
        image_format = b'png'
    else:
        print(f"Cảnh báo: Định dạng ảnh không xác định '{image_format_str}' cho tệp {filename}. Mặc định là jpeg.")
        image_format = b'jpeg'

    try:
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
    except Exception as e:
        print(f"Lỗi: Tệp XML bị hỏng (thiếu width/height): {xml_file}. Lỗi: {e}")
        return None

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label_map:
            print(f"Cảnh báo: class '{class_name}' trong {xml_file} không có trong label_map. Bỏ qua object này.")
            continue
            
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map[class_name])
        
        bndbox = obj.find('bndbox')
        xmins.append(float(bndbox.find('xmin').text) / width)
        xmaxs.append(float(bndbox.find('xmax').text) / width)
        ymins.append(float(bndbox.find('ymin').text) / height)
        ymaxs.append(float(bndbox.find('ymax').text) / height)
    
    
    if len(classes) == 0:
        print(f"Cảnh báo: Không tìm thấy object hợp lệ nào trong {xml_file}. Bỏ qua tệp này.")
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


print(f"Đang tìm tệp XML trong: {annotations_dir}")
xml_files = glob.glob(os.path.join(annotations_dir, '*.xml'))
print(f"Tìm thấy {len(xml_files)} tệp XML.")

if not xml_files:
    print(f"LỖI: Không tìm thấy tệp .xml nào trong {annotations_dir}.")
    print("Hãy đảm bảo bạn đã tạo thư mục 'Data/annotations_test' và di chuyển các tệp .xml vào đó.")
else:
    writer = tf.io.TFRecordWriter(test_record_path)
    count_success = 0
    for xml_file in xml_files:
        tf_example = create_tf_example(xml_file)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            count_success += 1
    
    writer.close()
    print(f'\nĐã tạo thành công tệp "Data/test.record".')
    print(f"Đã xử lý thành công {count_success} / {len(xml_files)} tệp XML.")
