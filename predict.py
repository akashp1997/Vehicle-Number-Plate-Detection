import tensorflow as tf
import numpy as np




sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('training/model.ckpt-500.meta')
saver.restore(sess,tf.train.latest_checkpoint('training/'))

graph = tf.get_default_graph()

input_node = tf.placeholder(np.uint8, shape = [1, 300, 300, 3])
output_node = [op for op in graph.get_operations() if "stack" in op.name]
print(output_node)
