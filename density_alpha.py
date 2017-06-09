# training score 14.?

from density_params import *


class Alpha(Params):
    model_name = 'alpha'

    model_dir = 'model_data/alpha/'

    density_dim = 7

    density_border = 1

    top_layers = [
        conv_layer(1024, (3, 3), name="block15_conv1024", padding='valid'),
        conv_layer(256, (1, 1), name="block15_conv256"),
        final_conv_layer(5, (1, 1), name="block15_conv5"),
        Activation('softplus', name='block15_conv5_act')
    ]

    predict_batch_size = 64

    @property
    def training_runs(self):
        return [TrainingRun(loss=self.area_loss,
                            optimizer=RMSprop(1e-3),
                            batch_size=64,
                            trainable=lambda layer: str.startswith(layer.name, "block15"),
                            keep_prob=reject_empty(0.05),
                            callbacks=[EarlyStopping('loss', 1e-4)],
                            metrics=[count_error]),
                TrainingRun(loss=self.area_loss,
                            optimizer=RMSprop(1e-5),
                            batch_size=64,
                            trainable=lambda layer: (str.startswith(layer.name, "block14") or
                                                        str.startswith(layer.name, "block15")),
                            keep_prob=reject_empty(1.0),
                            callbacks=[ReduceLROnPlateau('loss', patience=0),
                                        EarlyStopping('loss', 1e-6)],
                            metrics=[count_error])]


params = Alpha()
