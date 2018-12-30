from network.norm_type import NormType


class CFG:
    is_load_model = False
    cuda = False

    start_epoch = 1
    num_epoch = 100
    batch_size = 8

    lr = 2e-3
    betas = (0.5, 0.99)
    norm_type = NormType.Batch

    loss_span = 100
    save_result_span = 100
    save_model_span = 100

    # FIXME change path
    result_dir = 'path/to/result_dir'
    model_dir = 'path/to/model_dir'

    line_path = 'path/to/line_art_dir'
    color_path = 'path/to/color_art_dir'

    model_name = 'best.pth'
