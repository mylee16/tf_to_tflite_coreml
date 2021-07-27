import coremltools as ct


def tf_to_coreml(tf_model, mlmodel_save_path):
    mlmodel = ct.convert(
        tf_model, inputs=[ct.ImageType(bias=[-1, -1, -1], scale=1 / 127)]
    )
    mlmodel.save(mlmodel_save_path)
