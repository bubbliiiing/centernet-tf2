import os

import tensorflow as tf
from tqdm import tqdm


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

def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, save_period, save_dir):
    train_step  = get_train_step_fn()

    total_loss  = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_step:
                break
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch

            loss_value = train_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            total_loss += loss_value

            pbar.set_postfix(**{'total_loss'        : float(total_loss) / (iteration + 1), 
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration>=epoch_step_val:
                break
            # 计算验证集loss
            batch = [tf.convert_to_tensor(part) for part in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices = batch

            loss_value = val_step(batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, net, optimizer)
            val_loss += loss_value

            # 更新验证集loss
            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    logs = {'loss': total_loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
