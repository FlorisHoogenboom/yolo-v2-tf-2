# Yolo-v2-tf-2
This is a basic implementation of Yolo-v2 using Tensorflow 2 and eager execution. The purpose of this repository is not to provide the most performant model but rather to explore the ideas that underpin Yolo-v2 and subsequent versions.

## Motivation
The goal of this implementation is not to be the most performant implementation of this algorith. Rather, it is inteded to provide a simple overview of the mechanics involved in implementing an object detector using anchor boxes. Even though Yolo might not be the most complex network out there, the implementation of the anchor layer and loss computation are not trival.

## Usage
Clone this repository and in the root run ```pip install .```. This should set you up and make the `yolo` package available to you. To see how to use the model please refer to [this demo notebook](./Demo.ipynb).

When initialized the network is ofcourse untrained. Weights trained on Pascal VOC are available uppon request. __WARNING:__ These weights are intended for illustrational purposes. If you want to run this for quality of detections, please use the original weights as available on the yolo project page.

## Questions you'll probably have after browsing this repo
- __Why v2 and not v3?__: Yolo-v2 is relatively more simple than yolo-v3 since it only has a single anchor/detector head. However, this implementation is done in such a way that one could easilly add multiple anchor heads in the `yolo.model.Yolo` implementation (see [here](yolo/model.py#L221))
- __Is this performance wise comparable to the original Yolo implementation__: No. Don't use this if you care about quality. The goal of this repo is understanding.
-- __Which backbones are available__: Currently the darknet backbone is hardcoded. I am working on adding support for other backbones.
