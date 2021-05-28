from tensorflow import keras

class ScaledMSE(keras.metrics.MeanSquaredError):
    """Calculate the mean squared error with a specified scaling factor applied."""
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
    """Calculate the mean absolute error with a specific scaling factor applied."""
    def __init__(self, scale=1., *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def result(self):
        return super().result()*self.scale

    def get_config(self):
        result = super().get_config()
        result['scale'] = self.scale
        return result
