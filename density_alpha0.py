# training score 14.?

from density_params import *


class Alpha0(Params):
    model_name = 'alpha0'

    # where to store the weights and logs
    model_dir = 'model_data/alpha0/'

    # the number of rows/cols of density cells output by the NN
    density_dim = 7

    # The number of rows/cols of density cells discarded from the edges.
    # These increase the size of the input image, but not the size of the
    # output density
    density_border = 1

    # list of top layers
    top_layers = [
        conv_layer(1024, (3, 3), name="block15_conv1024", padding='valid'),
        conv_layer(256, (1, 1), name="block15_conv256"),
        final_conv_layer(5, (1, 1), name="block15_conv5"),
        Activation('softplus', name='block15_conv5_act')
    ]

    # The number of small (327 pixel) input images to batch together when
    # making predictions on a full size input image
    predict_batch_size = 64

    # The phases of training.
    # Here, we first train just the new layers to convergence, and then train
    # the last block of xception convolution layers, together with our new layers
    @property
    def training_runs(self):
        return [TrainingRun(loss=self.area_loss,
                            optimizer=RMSprop(1e-3),
                            batch_size=64,
                            trainable=lambda layer: str.startswith(layer.name, "block15"),
                            keep_prob=reject_empty(0.01),
                            callbacks=[EarlyStopping('loss', patience=10)],
                            metrics=[count_error]),
                TrainingRun(loss=self.area_loss,
                            optimizer=RMSprop(1e-5),
                            batch_size=32,
                            trainable=lambda layer: (str.startswith(layer.name, "block14") or
                                                        str.startswith(layer.name, "block15")),
                            keep_prob=reject_empty(1.0),
                            callbacks=[EarlyStopping('loss', patience=10)],
                            metrics=[count_error])]


params = Alpha0()
