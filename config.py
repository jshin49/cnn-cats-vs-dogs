class Config(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.001  		# Learning Rate
    l2 = 0.0001         # L2 Loss Coefficient
    dropout = 0.5     	# Dropout Rate
    batch_size = 32		# SGD Batch Size
    epochs = 10			# Number of Training Epochs
    threshold = 0.5     # Threshold for classification

    # Data Processing
    train_dir = './data/train'
    test_dir = './data/test'
    image_dir = './arrays/'
    image_size = 64     # resize image to image_size*image_size
    channels = 3        # Channel Size
    valid_size = 1000
    train_size = 25000 - valid_size

    # Saver
    model_name = 'cnn-tflayers-model-l2{}-dropout{}-batch{}-lr{}-img{}'.format(
        l2, dropout, batch_size, lr, image_size)
    ckpt_path = 'ckpt/' + model_name
