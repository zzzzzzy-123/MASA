from model.tcn_plus import MASA_TCN_Regressor


def get_model(features=None, num_chan=8, thickness=5, dropout=0.3, **kwargs):
    """Build the MASA TCN model used by the training loop.

    Dataset samples are shaped as [batch, 1, num_chan * thickness, time].
    MASA_TCN_Regressor expects the channel/frequency factors separately.
    """
    hidden_channels = kwargs.get("hidden_channels", [64, 64])
    kernel_sizes = kwargs.get("kernel_sizes", [2, 4, 6])
    max_si_score = kwargs.get("max_si_score", 30)

    return MASA_TCN_Regressor(
        num_channels_list=hidden_channels,
        num_eeg_chan=num_chan,
        freq=thickness,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        max_si_score=max_si_score,
    )
