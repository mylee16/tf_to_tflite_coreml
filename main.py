import tensorflow as tf
import numpy as np
from src.mobileunet import MaskMeanIoU, precision, recall, fbeta
from src.tf_to_tflite import tf_to_tflite
from src.tf_to_coreml import tf_to_coreml

if __name__ == "__main__":
    model_dir = "models/model.h5"
    tflite_save_path = "models/mobileunet_model.tflite"
    coreml_save_path = "models/mobileunet_model.mlmodel"

    tf_model = tf.keras.models.load_model(
        model_dir,
        custom_objects={
            "MaskMeanIoU": MaskMeanIoU,
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        },
    )

    tf_to_tflite(tf_model, tflite_save_path)
    tf_to_coreml(tf_model, coreml_save_path)
    print("done")
