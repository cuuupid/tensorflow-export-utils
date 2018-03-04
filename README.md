# Freezing Models

## Preparation

Download the checkpoint files to `ckpt` folder here. (Use the MobilenetV1 224 pretrained checkpoint from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained))

Export the inference graph:
``` bash
python export_inference_graph.py \
    --alsologtostderr --model_name=mobilenet_v1 \
    --image_size=224 --output_file=mobilenet.pb
```

Freeze the graph:
``` bash
python freeze_graph.py --input_graph=mobilenet.pb \
                       --input_checkpoint=ckpt/mobilenet_v1_1.0_224.ckpt \
                       --output_graph=mobilenet_f.pb --input_binary=true \
                       --output_node_names=MobilenetV1/Predictions/Reshape_1
```

Edit `pb_to_pbtxt.py` to use the name of your PB file as graph.
Make pbtxt:
``` bash
python pb_to_pbtxt.py
```

## Serving

See `api.py` for example. Change input and output tensor names accordingly.

## Credits

Got the export and freezing scripts from tensorflow. Basically moving the utilities out of the repo to allow them to be used without bazel.

Models from tensorflow/models repo.

Datasets, pretrained checkpoints, etc. from tensorflow/models/research/slim.

**Working as of 3/4/2018.**
