import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.layers.core import Lambda
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from utils.networks.resnet50_2classes import ResNet
import os
from data_enhancement import BinarySP
K.set_learning_phase(0)
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))
  
def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):#对张量进行归一化
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img = cv2.imread(path)
    img = np.float32(img)
    img = (img/255 - 0.5)*2
    img = np.expand_dims(img, axis=0)
    return img
def load_image_BinarySP(path):
    img = cv2.imread(path)
    imgBin = BinarySP(img.copy(),28,1000)
    img = np.float32(img)
    img = (img/255 - 0.5)*2
    img [imgBin==0] = 0
    img = np.expand_dims(img, axis=0)
    return img

def grad_cam(input_model, image, category_index, layer_name):
    x = input_model.output
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    y = Lambda(target_layer, output_shape = target_category_loss_output_shape)(x)
    model = Model(input_model.input, y)
    
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name==layer_name][0].output#获得该图片的特征图
    x=K.gradients(loss, conv_output)[0]
    grads = normalize(x)
    gradient_function = K.function([model.input], [conv_output,grads])
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (512, 512))
    cam = np.maximum(cam, 0)#即最小值为0
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image = np.uint8((image/2+0.5)*255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam), heatmap


fold = ""
gradcam_path= ""#
listDir=os.listdir(fold)
input_model = ResNet(input_shape=(512,512,3))
input_model.load_weights(".h5")

classes = ["activation_10","activation_22","activation_40","activation_49"]
font=cv2.FONT_HERSHEY_SIMPLEX
for cz in listDir:
    print(cz)
    path=fold+cz
    img = load_image(path)
    imgcam = cv2.imread(path)
    predictions = input_model.predict(img)
    predicted_class = np.argmax(predictions)
    for item in classes:
        cam, heatmap = grad_cam(input_model, img, predicted_class, item)
        cam = cv2.putText(cam,str(item),(100,50),font,1.5,(0,255,0),3)
        imgcam = np.concatenate((imgcam,cam),axis = 1)
    cv2.imwrite(gradcam_path+cz.split(".tif")[0]+str(predictions)+"_gradcam.jpg",imgcam)
       