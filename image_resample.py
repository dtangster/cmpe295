import os

import tensorflow as tf


def resample(filename, scale=1.0, method=tf.image.ResizeMethod.BICUBIC, output_filename=None):
    if not output_filename:
        print(filename)
        basename, ext = filename.rsplit(".")
        output_filename = f"{basename}_scale_{scale}.{ext}"

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    file_content = tf.io.read_file(filename)
    try:
        image = tf.image.decode_image(file_content)
    except Exception:
        print("Invalid image: ", filename)
        return

    new_size = tf.constant([image.shape[0], image.shape[1]], dtype=tf.float64)
    new_size *= scale
    new_size = tf.dtypes.cast(new_size, tf.int32)

    try:
        resized_image = tf.image.resize(image, size=new_size, method=method, preserve_aspect_ratio=True)
    except Exception:
        print("Failed to resize: %s, %r", filename, image.shape)
        return

    try:
        tf.keras.preprocessing.image.save_img(output_filename, resized_image)
    except Exception:
        print("Failed to save: ", output_filename)

folder = "Andrew Lincoln"
lr_folder = folder + " LRx4"

for filename in os.listdir(folder):
    src_path = os.path.join(folder, filename)
    dst_path = os.path.join(lr_folder, filename)
    resample(src_path, scale=0.25, output_filename=dst_path)
