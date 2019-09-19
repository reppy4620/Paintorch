class CFG:
    is_load_model = False
    cuda = True

    start_epoch = 1
    num_epoch = 100
    batch_size = 8

    lr = 2e-3
    betas = (0.5, 0.99)

    save_result_span = 100
    save_model_span = 100

    # FIXME change path
    result_dir = 'D:/result_path'
    model_dir = 'D:/model_path'

    color_path = 'D:/ImageDatas/color_path'

    model_name = 'best.pth'
