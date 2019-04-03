from keras.models import load_model, Model, Input
from keras.utils.vis_utils import plot_model 
from keras import backend as K
import tensorflow as tf

IMG_H, IMG_W = 406, 528
TARGET_IMG_H, TARGET_IMG_W = 416, 544
BATCH_SIZE = 5
LAYERS = ["add_19", "conv2d_75"]

imgs = tf.ones([BATCH_SIZE, TARGET_IMG_H, TARGET_IMG_W, 3])

with tf.Session() as sess:
    K.set_learning_phase(0)
    model = load_model("yolov3.h5", compile=False)
    print("plot model")
    plot_model(model, to_file="model.png", show_shapes=True)

    print(f"find {len(LAYERS)} in {len(model.layers)} model layers")
    for i, layer in enumerate(model.layers):
        if layer.name in LAYERS:
            print(f"found layer '{layer.name}' at index: {i}")

    print("eval layer dimensions")
    outputs = [model.get_layer(l).output for l in LAYERS]
    model = Model(model.input, outputs)
    out = sess.run(model(imgs))
    out = [f"{layer}: {output.shape}" for (layer, output) in zip(LAYERS, out)]

    print(f"using BATCH_SIZE of {BATCH_SIZE}")
    print("\n".join(out))
