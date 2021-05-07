from tensorflow import keras

class ScaledMSE(keras.metrics.MeanSquaredError):
    def __init__(self, scale=1., *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def result(self):
        return super().result()*self.scale**2

    def get_config(self):
        result = super().get_config()
        result['scale'] = self.scale
        return result

class ScaledMAE(keras.metrics.MeanAbsoluteError):
    def __init__(self, scale=1., *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def result(self):
        return super().result()*self.scale

    def get_config(self):
        result = super().get_config()
        result['scale'] = self.scale
        return result
