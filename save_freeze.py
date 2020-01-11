import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('training/model.ckpt-500.meta')
saver.restore(sess,tf.train.latest_checkpoint('training/'))

graph = tf.get_default_graph()
print(graph.get_operations())
quit()
input_graph_def = graph.as_graph_def()
output_node_names = ['stack_12']

print(graph.get_operation_by_name(output_node_names[0]))

output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

with tf.gfile.GFile("training/frozen_inference_graph.pb", 'wb') as f:
    f.write(output_graph_def.SerializeToString())
