"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from dao import Dao
import time
import os.path

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    train_data = Dao(
            './train_dataset_desc', 
            seq_length=seq_length,
            image_shape=image_shape
        )
    validation_data = Dao(
            './validation_dataset_desc', 
            seq_length=seq_length,
            image_shape=image_shape
        )
        
    steps_per_epoch = train_data.size() // batch_size

    train_gen = train_data.frame_generator(batch_size)
    val_generator = validation_data.frame_generator(batch_size)
        

    # Get the model.
    rm = ResearchModels(train_data.num_of_classes(), model, seq_length, saved_model)

    rm.model.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=40,
        workers=4)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lrcn'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    batch_size = 32
    nb_epoch = 20

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
