import os, shutil
import tensorflow as tf
import Data_loader
import Unet_model




if __name__ == '__main__':
    channel_lst = [64,128,256,512,1024]
    model = Unet_model.build_UNET(channel_lst,(512,512,1))
    trn_dl,val_dl,tst_dl = Data_loader.data_loader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0)
    model.compile(optimizer=optimizer, loss=loss)
    model_dir = './model_save/CNN/sigmoid_interp_length_Res5class_multilabel_age_sex__0003'
    try:
        shutil.mkdir(model_dir)
    except:
        pass
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        save_weights_only=False,
        monitor='val_roc',
        mode='max',
        save_best_only=True)
    model_earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    history = model.fit(trn_dl, validation_data=val_dl, epochs=1000,
                        callbacks=[model_checkpoint_callback, model_earlystopping_callback])
