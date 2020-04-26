import tensorflow as tf

def smooth_l1(x):

    def func1():
        return x**2 * 0.5

    def func2():
        return tf.abs(x) - tf.constant(0.5)

    def f(x): return tf.cond(tf.less(tf.abs(x), tf.constant(1.0)), func1, func2)

    return tf.map_fn(f, x)

#loss implemented as the smooth l1 loss in SSD.
#dimensions of pred : [batch_size, 320, 5] where the last dimension corresponds to score ad the 4 predicted location values.
#second dimentions (320) corresponds to the total feature points (16*16 + 8*8) for to the two feature maps of sizes 16*16 and 8*8. 
def smooth_l1_loss(true, pred):



    face_label = true[:, :, :1]
    #target center x coordinate. 
    gxs = true[:, :, 1:2]
    #target center y coordinate. 
    gys = true[:, :, 2:3]
    #target width 
    gws = true[:, :, 3:4]
    #target height
    ghs = true[:, :, 4:5]

    dxs = true[:, :, 5:6]
    dys = true[:, :, 6:7]
    dws = true[:, :, 7:8]
    dhs = true[:, : ,8:9]

    pxs = pred[:, :, 1:2]
    pys = pred[:, :, 2:3]
    pws = pred[:, :, 3:4]
    phs = pred[:, :, 4:5]

    logitx = (gxs - dxs) / dws
    logity = (gys - dys) / dhs
    logitw = tf.math.log(gws / dws)
    logith = tf.math.log(ghs/ dhs)

    lossx = face_label * tf.map_fn(smooth_l1, tf.reshape(pxs - logitx, (-1, 320)))
    lossy = face_label * tf.map_fn(smooth_l1, tf.reshape(pys - logity, (-1, 320)))
    lossw = face_label * tf.map_fn(smooth_l1, tf.reshape(pws - logitw, (-1, 320)))
    lossh = face_label * tf.map_fn(smooth_l1, tf.reshape(phs - logith, (-1, 320)))

    x_sum = tf.reduce_sum(lossx)
    y_sum = tf.reduce_sum(lossy)
    w_sum = tf.reduce_sum(lossw)
    h_sum = tf.reduce_sum(lossh)

    loss = tf.stack((x_sum, y_sum, w_sum, h_sum))

    return tf.reduce_mean(loss)


if __name__ == "__main__":
    pass