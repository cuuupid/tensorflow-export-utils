import tensorflow as tf
graph_filename = "./mobilenet_f.pb"

with tf.gfile.GFile(graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.train.write_graph(graph_def, './', 'mobilenet_f.pbtxt', True)