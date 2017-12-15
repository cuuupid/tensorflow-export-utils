# Mobilenet

## Preparation

Download the checkpoint files to `ckpt` folder here. (Use the MobilenetV1 224 pretrained checkpoint from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained))

Export the inference graph:
``` bash
python export_inference_graph.py --alsologtostderr --model_name=mobilenet_v1 --image_size=224 --output_file=mobilenet.pb
```

Freeze the graph:
``` bash
python freeze_graph.py --input_graph=mobilenet.pb --input_checkpoint=ckpt/mobilenet_v1_1.0_224.ckpt --output_graph=mobilenet_f.pb --input_binary=true --output_node_names=MobilenetV1/Predictions/Reshape_1
```

Make pbtxt:
``` bash
python pb_to_pbtxt.py
```