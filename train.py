from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tqdm import tqdm

from nets.centernet import centernet
from nets.centernet_training import Generator, LossHistory
from utils.utils import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer):
        with tf.GradientTape() as tape:
            loss_value = net([batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], training=True)
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

@tf.function
def val_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer):
    loss_value = net([batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices])
    return loss_value

def fit_one_epoch(net, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, train_step=None):

    total_loss = 0
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch

            loss_value = train_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            total_loss += loss_value

            pbar.set_postfix(**{'total_loss'        : float(total_loss) / (iteration + 1), 
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch

            loss_value = val_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            val_loss += loss_value

            # 更新验证集loss
            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss.numpy()/(epoch_size+1), 'val_loss': val_loss.numpy()/(epoch_size_val+1)}
    loss_history.on_epoch_end([], logs)
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__": 
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager = False
    #-----------------------------#
    #   图片的大小
    #-----------------------------#
    input_shape = [512,512,3]
    #-----------------------------#
    #   训练前一定要注意修改
    #   classes_path对应的txt的内容
    #   修改成自己需要分的类
    #-----------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #----------------------------------------------------#
    #   获取classes和数量
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    #-----------------------------#
    #   主干特征提取网络的选择
    #   resnet50
    #   hourglass
    #-----------------------------#
    backbone = "resnet50"

    #----------------------------------------------------#
    #   获取centernet模型
    #----------------------------------------------------#
    model = centernet(input_shape, num_classes=num_classes, backbone=backbone, mode='train')
    
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = r"model_data/centernet_resnet50_voc.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)

    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir="logs/")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)
    loss_history = LossHistory("logs/")

    if backbone == "resnet50":
        freeze_layer = 171
    elif backbone == "hourglass":
        freeze_layer = 624
    else:
        raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(backbone))

    for i in range(freeze_layer):
        model.layers[i].trainable = False

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        Lr              = 1e-3
        Batch_size      = 4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        if eager:
            generator       = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)
            
            gen             = tf.data.Dataset.from_generator(partial(generator.generate, train = True, eager = True), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val         = tf.data.Dataset.from_generator(partial(generator.generate, train = False, eager = True), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

            gen             = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val         = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

            lr_schedule     = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=Lr, decay_steps=epoch_size, decay_rate=0.92, staircase=True
            )
            optimizer       = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(Init_Epoch,Freeze_Epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, get_train_step_fn())

        else:
            gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)
            model.compile(
                loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                optimizer=keras.optimizers.Adam(Lr)
            )
            model.fit(gen.generate(True), 
                    steps_per_epoch=epoch_size,
                    validation_data=gen.generate(False),
                    validation_steps=epoch_size_val,
                    epochs=Freeze_Epoch, 
                    verbose=1,
                    initial_epoch=Init_Epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])


    for i in range(freeze_layer):
        model.layers[i].trainable = True

    if True:
        Lr              = 1e-4
        Batch_size      = 4
        Freeze_Epoch    = 50
        Epoch           = 100
        
        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        if eager:
            generator       = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)
            
            gen             = tf.data.Dataset.from_generator(partial(generator.generate, train = True, eager = True), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val         = tf.data.Dataset.from_generator(partial(generator.generate, train = False, eager = True), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

            gen             = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val         = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            
            lr_schedule     = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=Lr, decay_steps=epoch_size, decay_rate=0.92, staircase=True
            )

            optimizer       = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
            for epoch in range(Freeze_Epoch,Epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, get_train_step_fn())

        else:
            gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)
            model.compile(
                loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                optimizer=keras.optimizers.Adam(Lr)
            )
            model.fit(gen.generate(True), 
                    steps_per_epoch=epoch_size,
                    validation_data=gen.generate(False),
                    validation_steps=epoch_size_val,
                    epochs=Epoch, 
                    verbose=1,
                    initial_epoch=Freeze_Epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
