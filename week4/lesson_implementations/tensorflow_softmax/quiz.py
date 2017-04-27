# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    softmax = tf.nn.softmax(logits=logits)

    output = None
    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output
