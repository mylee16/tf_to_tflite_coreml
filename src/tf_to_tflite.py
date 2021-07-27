import tensorflow as tf
import pathlib


def tf_to_tflite(tf_model, tflite_save_path):

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    tflite_model = converter.convert()

    tflite_model_file = pathlib.Path(tflite_save_path)
    tflite_model_file.write_bytes(tflite_model)
