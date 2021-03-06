class Config(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.001  		# Learning Rate
    dropout = 0.5  		# Dropout Rate
    batch_size = 16		# SGD Batch Size
    epochs = 10			# Number of Training Epochs

    # Data Processing
    train_dir = './data/train'
    test_dir = './data/test'
    image_size = 150    # resize image to image_size*image_size
    channels = 3        # Channel Size
    split_rate = 0.2 	# Validation Data portion
    valid_size = int(25000 * split_rate)
    train_size = 25000 - valid_size

    # Saver
    model_name = 'full-{}lr-{}img'.format(lr, image_size)
    ckpt_path = 'ckpt/' + model_name
