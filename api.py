import tensorflow as tf
from scipy.misc import imread, imresize
from time import time as t
import numpy as np


def get_labels(labels_filename):
    labels_dict = {}
    with open(labels_filename, 'r') as f:
        for kv in [d.strip().split(':') for d in f]:
            labels_dict[int(kv[0])] = kv[1]
    return labels_dict


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="graph")
    return graph


def process(image_filename, graph, labels_dict):
    img = imread(image_filename)
    img = imresize(img, (224, 224, 3))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = (img / 255) - 0.5 * 2.
    x = graph.get_tensor_by_name('graph/input:0')
    y = graph.get_tensor_by_name('graph/MobilenetV1/Predictions/Reshape:0')

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(y, feed_dict={x: img})[0]
        preds = predictions.argsort()[-5:][::-1]
        probs = predictions[preds]
        classes = [labels_dict[i] for i in preds]
        return list(zip(classes, probs))


def test(filename, graph, labels):
    start = t()
    preds = process(filename, graph, labels)
    print("Finished in %f ms" % (t() - start))
    print("Filename: %s" % filename.split('.')[0])
    for i, pred in enumerate(preds):
        print(str(i) + " | %s : %s" % pred)
    print('-' * 20)
