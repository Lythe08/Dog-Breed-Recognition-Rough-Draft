#Dog breed recognition model
#!/usr/bin/env/python3

#importing jetson modules and argument parser
import jetson_inference
import jetson_utils
import argparse

#parses image file name and selects network
parser = argparse.ArgumentParser() 
parser.add_argument("filename", type = str, help = "filename of the image to process") 
parser.add_argument("--network", type = str, default = "googlenet", help = "model to use, googlenet is default") 
opt = parser.parse_args()


#loads image
img = jetson_utils.loadImage(opt.filename)

#load recognition network from command line
net = jetson_inference.imageNet(model="dog_breeds/resnet18.onnx", labels="dog_breeds/labels.txt", input_blob="input_0", output_blob="output_0")

#class_idx - index of predicted class
#confidence - accuracy of prediction
class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as " + str(class_desc) + "(class #" + str(class_idx) + ") with confidence: " + str(confidence*100) + "%")