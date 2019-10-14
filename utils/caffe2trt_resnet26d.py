import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import model
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import numpy as np
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 1000
    DTYPE = trt.float32

def layer_conv(network, input=None, weight=None, output_size=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=None):
    conv = network.add_convolution(input,
                                   num_output_maps=output_size,
                                   kernel_shape=kernel_size,
                                   kernel=weight,
                                   bias=np.zeros(output_size).astype(np.float32))
    conv.stride = stride

    #if padding_mode != None:
    #    conv.padding_mode = padding_mode
    #print(padding_mode)
    conv.padding = padding
    #conv.pre_padding = padding
    #conv.post_padding = padding
    '''
    DEFAULT             :                                                 # 112 55 27/28
    EXPLICIT_ROUND_DOWN : Use explicit padding, rounding the output size down
    EXPLICIT_ROUND_UP : Use explicit padding, rounding the output size up # 113 56 28/29
    SAME_UPPER : Use SAME padding, with pre_padding <= post_padding
    SAME_LOWER : Use SAME padding, with pre_padding >= post_padding
    CAFFE_ROUND_DOWN : Use CAFFE padding, rounding the output size down
    CAFFE_ROUND_UP : Use CAFFE padding, rounding the output size up
    '''
    return conv

def layer_bn(network, input, g0, b0, m0, v0 ):
    ''''
    adjustedScale = scale / sqrt(variance + epsilon)
    batchNorm = (input + bias - (adjustedScale * mean)) * adjustedScale
    '''
    #g0 = params['batchnorm0_gamma'].asnumpy().reshape(-1)
    #b0 = params['batchnorm0_beta'].asnumpy().reshape(-1)
    #m0 = extra_params['batchnorm0_moving_mean'].asnumpy().reshape(-1)
    #v0 = extra_params['batchnorm0_moving_var'].asnumpy().reshape(-1)
    scale0 = g0 / np.sqrt(v0 + 2e-5)
    shift0 = -m0 / np.sqrt(v0 + 2e-5) * g0 + b0
    power0 = np.ones(len(g0), dtype=np.float32)
    batchNormLayer = network.add_scale(input, trt.ScaleMode.CHANNEL,
                                       trt.Weights(shift0), trt.Weights(scale0), trt.Weights(power0))
    return batchNormLayer

def build_block(network, input, layer_list, weights, weight_list, bn_param_list, output_size_list,
                kernel_size_list, stride_list, avgpool_kernel_size=0, avgpool_stride_size=0, padding_mode=None):
    output = input
    conv_count=0
    bn_count =0
    for l in layer_list:
        if l == 'Conv2d':
            #output_layer = network.add_padding(input=output,
            #                                   pre_padding=(int(kernel_size_list[conv_count]/2), int(kernel_size_list[conv_count]/2)),
            #                                   post_padding=((int(kernel_size_list[conv_count]/2), int(kernel_size_list[conv_count]/2))))
            #_layer.get_output(0)
            output_layer = layer_conv(network, output,
                                      weight=weights[weight_list[conv_count]].numpy(),
                                      output_size=output_size_list[conv_count],
                                      kernel_size=(kernel_size_list[conv_count], kernel_size_list[conv_count]),
                                      stride=(stride_list[conv_count], stride_list[conv_count]),
                                      padding= (int(kernel_size_list[conv_count]/2), int(kernel_size_list[conv_count]/2)),
                                      padding_mode=padding_mode)
            conv_count+=1
            print('AFTER CONVOLUTION')
            print('output_tensor shape: ', output_layer.get_output(0).shape)  # output_layer.get_output(0).shape[1])
            print('kernel :', (kernel_size_list[conv_count - 1], kernel_size_list[conv_count - 1]), 'padding : ',
                  (int(kernel_size_list[conv_count - 1] / 2), int(kernel_size_list[conv_count - 1] / 2)), 'stride: ',
                  (stride_list[conv_count - 1], stride_list[conv_count - 1]))

        elif l == 'BatchNorm2d':
            g=weights[bn_param_list[bn_count][0]].numpy()
            b=weights[bn_param_list[bn_count][1]].numpy()
            m=weights[bn_param_list[bn_count][2]].numpy()
            v=weights[bn_param_list[bn_count][3]].numpy()
            output_layer = layer_bn(network, output, g, b, m, v)
            bn_count += 1
        elif l == 'Relu':
            output_layer = network.add_activation(input=output, type=trt.ActivationType.RELU)
        elif l == 'AvgPool2d':
            output_layer = network.add_pooling(output, trt.PoolingType.AVERAGE, (avgpool_kernel_size, avgpool_kernel_size))
            output_layer.stride = (avgpool_stride_size, avgpool_stride_size)
            output_layer.padding = (0, 0)
            print('AFTER AVG POOL')
            print('output_tensor shape: ', output_layer.get_output(0).shape)  # output_layer.get_output(0).shape[1])
            print('kernel :', (kernel_size_list[conv_count - 1], kernel_size_list[conv_count - 1]), 'padding : ',
                  (int(kernel_size_list[conv_count - 1] / 2), int(kernel_size_list[conv_count - 1] / 2)), 'stride: ',
                  (stride_list[conv_count - 1], stride_list[conv_count - 1]))
        output = output_layer.get_output(0)
        print('')

    return output
def build_layer(network,weights, input_layer, layer_number, output_size_lists, stride_size_lists, padding_mode):
    '''
        (0): Bottleneck(
        # block1_0
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)                           6
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block1_1
          (downsample): Sequential(
            (0): AvgPool2d(kernel_size=1, stride=1, padding=0)------------------1
            (1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)                             7
            (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )

          ADD(block0_0 , block0_1)=>(relu): ReLU(inplace=True) => OUTPUT

        (1): Bottleneck(
        # block2
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)                           10
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ) => OUPUT'

        ADD(OUTPUT , OUTPUT') => (relu): ReLU(inplace=True)
        '''
    layer_list = ['Conv2d', 'BatchNorm2d',
                  'Conv2d', 'BatchNorm2d',
                  'Conv2d', 'BatchNorm2d']

    weight_key_list = ['layer'+str(layer_number)+'.0.conv1.weight', 'layer'+str(layer_number)+'.0.conv2.weight', 'layer'+str(layer_number)+'.0.conv3.weight']
    bn_param_key_list = [
        ['layer'+str(layer_number)+'.0.bn1.weight', 'layer'+str(layer_number)+'.0.bn1.bias', 'layer'+str(layer_number)+'.0.bn1.running_mean', 'layer'+str(layer_number)+'.0.bn1.running_var'],
        ['layer'+str(layer_number)+'.0.bn2.weight', 'layer'+str(layer_number)+'.0.bn2.bias', 'layer'+str(layer_number)+'.0.bn2.running_mean', 'layer'+str(layer_number)+'.0.bn2.running_var'],
        ['layer'+str(layer_number)+'.0.bn3.weight', 'layer'+str(layer_number)+'.0.bn3.bias', 'layer'+str(layer_number)+'.0.bn3.running_mean', 'layer'+str(layer_number)+'.0.bn3.running_var']]
    output_size_list = output_size_lists[0]
    kernel_size_list = [1, 3, 1]
    stride_list = stride_size_lists[0]
    block1_0_output_tensor = build_block(network, input_layer.get_output(0), layer_list, weights, weight_key_list,
                                         bn_param_key_list, output_size_list, kernel_size_list, stride_list, 0, 0, padding_mode=padding_mode)
    print('block1_0_output_tensor shape: ',block1_0_output_tensor.shape)

    layer_list = ['AvgPool2d', 'Conv2d', 'BatchNorm2d']
    weight_key_list = ['layer'+str(layer_number)+'.0.downsample.1.weight']
    bn_param_key_list = [
        ['layer'+str(layer_number)+'.0.downsample.2.weight',
         'layer'+str(layer_number)+'.0.downsample.2.bias',
         'layer'+str(layer_number)+'.0.downsample.2.running_mean',
         'layer'+str(layer_number)+'.0.downsample.2.running_var']]
    output_size_list = output_size_lists[1]
    kernel_size_list = [1]
    stride_list = stride_size_lists[1]
    if layer_number == 1:
        avg_pool_kernel_size=1
        avg_pool_stride_size=1
    else:
        avg_pool_kernel_size = 2
        avg_pool_stride_size = 2
    block1_1_output_tensor = build_block(network, input_layer.get_output(0), layer_list, weights, weight_key_list,
                                         bn_param_key_list, output_size_list, kernel_size_list, stride_list,
                                         avg_pool_kernel_size, avg_pool_stride_size, padding_mode=padding_mode)
    print('block1_1_output_tensor shape: ', block1_0_output_tensor.shape)  # output_layer.get_output(0).shape[1])

    add1 = network.add_elementwise(block1_0_output_tensor, block1_1_output_tensor, trt.ElementWiseOperation.SUM)
    assert add1 != None
    add1.get_output(0).name = 'Layer_'+str(layer_number)+' Block0 + Block1'
    block1_output_layer = network.add_activation(input=add1.get_output(0), type=trt.ActivationType.RELU)

    print('block1_output_layer shape: ', block1_output_layer.get_output(0).shape)

    layer_list = ['Conv2d', 'BatchNorm2d',
                  'Conv2d', 'BatchNorm2d',
                  'Conv2d', 'BatchNorm2d']
    weight_key_list = ['layer'+str(layer_number)+'.1.conv1.weight', 'layer'+str(layer_number)+'.1.conv2.weight', 'layer'+str(layer_number)+'.1.conv3.weight']
    bn_param_key_list = [
        ['layer'+str(layer_number)+'.1.bn1.weight', 'layer'+str(layer_number)+'.1.bn1.bias', 'layer'+str(layer_number)+'.1.bn1.running_mean', 'layer'+str(layer_number)+'.1.bn1.running_var'],
        ['layer'+str(layer_number)+'.1.bn2.weight', 'layer'+str(layer_number)+'.1.bn2.bias', 'layer'+str(layer_number)+'.1.bn2.running_mean', 'layer'+str(layer_number)+'.1.bn2.running_var'],
        ['layer'+str(layer_number)+'.1.bn3.weight', 'layer'+str(layer_number)+'.1.bn3.bias', 'layer'+str(layer_number)+'.1.bn3.running_mean', 'layer'+str(layer_number)+'.1.bn3.running_var']]
    output_size_list = output_size_lists[2]
    kernel_size_list = [1, 3, 1]
    stride_list = stride_size_lists[2]
    block2_output_tensor = build_block(network, block1_output_layer.get_output(0), layer_list, weights, weight_key_list,
                                       bn_param_key_list, output_size_list, kernel_size_list, stride_list, 0, 0, padding_mode)
    print('block2_output_tensor shape: ', block2_output_tensor.shape)
    add2 = network.add_elementwise(block1_output_layer.get_output(0), block2_output_tensor,
                                   trt.ElementWiseOperation.SUM)
    assert add2 != None
    add2.get_output(0).name = 'Layer_'+str(layer_number)+' Block1 + Block2'
    block2_output_layer = network.add_activation(input=add2.get_output(0), type=trt.ActivationType.RELU)
    print('block2_output_layer shape: ', block2_output_layer.get_output(0).shape)
    return block2_output_layer

def populate_network_resnet26d(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # 1. STEM
    '''
    (conv1): Sequential((0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)------------------1 224->112
                        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (2): ReLU(inplace=True)
                        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)                 2
                        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (5): ReLU(inplace=True)
                        (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)                 3)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (relu): ReLU(inplace=True)
    '''
    layer_list = ['Conv2d', 'BatchNorm2d', 'ReLU',
                  'Conv2d', 'BatchNorm2d', 'ReLU',
                  'Conv2d', 'BatchNorm2d', 'ReLU']
    weight_key_list = ['conv1.0.weight',  'conv1.3.weight', 'conv1.6.weight']
    bn_param_key_list=[['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var'],
                       ['conv1.4.weight', 'conv1.4.bias', 'conv1.4.running_mean', 'conv1.4.running_var'],
                       ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']]
    output_size_list=[32,32,64]
    kernel_size_list=[3, 3, 3]
    stride_list=[2, 1, 1]
    stem_output_tensor=build_block(network, input_tensor, layer_list, weights, weight_key_list, bn_param_key_list, output_size_list, kernel_size_list, stride_list)
    #(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)----------------- 112->56
    maxpool_layer = network.add_pooling(stem_output_tensor, trt.PoolingType.MAX, (3, 3))
    maxpool_layer.stride = (2, 2)
    maxpool_layer.padding =(1, 1)
    print('maxpool_layer shape: ', maxpool_layer.get_output(0).shape)
    # 2. Layer 1~4
    output_layer1 = build_layer(network, weights, maxpool_layer,1,[[ 64, 64,256],  [256], [ 64, 64,256]], [[1, 1, 1], [1], [1, 1, 1]],None)#,trt.PaddingMode.SAME_UPPER None
    output_layer2 = build_layer(network, weights, output_layer1, 2,[[128,128,512],  [512], [128,128,512]], [[1, 2, 1], [1], [1, 1, 1]],None)#,trt.PaddingMode.EXPLICIT_ROUND_DOWN)
    output_layer3 = build_layer(network, weights, output_layer2, 3,[[256,256,1024],[1024],[256,256,1024]], [[1, 2, 1], [1], [1, 1, 1]],None)#,trt.PaddingMode.CAFFE_ROUND_UP)
    output_layer4 = build_layer(network, weights, output_layer3, 4,[[512,512,2048],[2048],[512,512,2048]], [[1, 2, 1], [1], [1, 1, 1]],None)#,trt.PaddingMode.SAME_UPPER)

    # 2. Layer 2

    # 'layer2.0.conv1.weight',
    # 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var',
    # 'layer2.0.conv2.weight',
    # 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
    # 'layer2.0.conv3.weight',
    # 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var',
    # 'layer2.0.downsample.1.weight',
    # 'layer2.0.downsample.2.weight', 'layer2.0.downsample.2.bias', 'layer2.0.downsample.2.running_mean', 'layer2.0.downsample.2.running_var',
    # 'layer2.1.conv1.weight',
    # 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var',
    # 'layer2.1.conv2.weight',
    # 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var',
    # 'layer2.1.conv3.weight',
    # 'layer2.1.bn3.weight', 'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var',

    # 'layer3.0.conv1.weight',
    # 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var',
    # 'layer3.0.conv2.weight',
    # 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
    # 'layer3.0.conv3.weight',
    # 'layer3.0.bn3.weight', 'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var',
    # 'layer3.0.downsample.1.weight',
    # 'layer3.0.downsample.2.weight', 'layer3.0.downsample.2.bias', 'layer3.0.downsample.2.running_mean', 'layer3.0.downsample.2.running_var',
    # 'layer3.1.conv1.weight',
    # 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
    # 'layer3.1.conv2.weight',
    # 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var',
    # 'layer3.1.conv3.weight',
    # 'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var',


    # 'layer4.0.conv1.weight',
    # 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var',
    # 'layer4.0.conv2.weight',
    # 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var',
    # 'layer4.0.conv3.weight',
    # 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var',
    # 'layer4.0.downsample.1.weight',
    # 'layer4.0.downsample.2.weight', 'layer4.0.downsample.2.bias', 'layer4.0.downsample.2.running_mean', 'layer4.0.downsample.2.running_var',
    # 'layer4.1.conv1.weight',
    # 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var',
    # 'layer4.1.conv2.weight',
    # 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var',
    # 'layer4.1.conv3.weight',
    # 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var',
    # 'fc.weight', 'fc.bias'])
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)

    output_layer5 = network.add_pooling(output_layer4.get_output(0), trt.PoolingType.AVERAGE, (7, 7))
    output_layer5.stride = (1, 1)
    output_layer5.padding = (0, 0)

    fc_w = weights['fc.weight'].numpy()
    fc_b = weights['fc.bias'].numpy()
    fc   = network.add_fully_connected(output_layer5.get_output(0), ModelData.OUTPUT_SIZE, fc_w, fc_b)

    fc.get_output(0).name = ModelData.OUTPUT_NAME
    print('FC LAYER shape: ', fc.get_output(0).shape)
    network.mark_output(tensor=fc.get_output(0))
'''
def tmp():
    #(relu): ReLU(inplace=True)

    #layer1-BOTTLENECK0
    #(conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer1_bottleneck0_conv1 = layer_conv(network, maxpool.get_output(0), weights['layer1.0.conv1.weight'].numpy(), 64, kernel_size=(1, 1), stride=(1, 1))
    #(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck0_bn1 = layer_bn(network, layer1_bottleneck0_conv1.get_output(0), weights['layer1.0.bn1.weight'].numpy(),
                                                                         weights['layer1.0.bn1.bias'].numpy(),
                                                                         weights['layer1.0.bn1.running_mean'].numpy(),
                                                                         weights['layer1.0.bn1.running_var'].numpy())
    #(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    layer1_bottleneck0_conv2_w = weights['layer1.0.conv2.weight'].numpy()
    layer1_bottleneck0_conv2 = network.add_convolution(input=layer1_bottleneck0_bn1.get_output(0), num_output_maps=64,
                                                       kernel_shape=(3, 3),
                                                       kernel=layer1_bottleneck0_conv2_w,
    #(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck0_bn2 = layer_bn(network, layer1_bottleneck0_conv2.get_output(0), weights['layer1.0.bn2.weight'].numpy(),
                                                                         weights['layer1.0.bn2.bias'].numpy(),
                                                                         weights['layer1.0.bn2.running_mean'].numpy(),
                                                                         weights['layer1.0.bn2.running_var'].numpy())
    #(conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer1_bottleneck0_conv3_w = weights['layer1.0.conv3.weight'].numpy()
    layer1_bottleneck0_conv3 = network.add_convolution(input=layer1_bottleneck0_bn2.get_output(0),
    #(bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck0_bn3 = layer_bn(network, layer1_bottleneck0_conv3.get_output(0), weights['layer1.0.bn3.weight'].numpy()
                                                                       , weights['layer1.0.bn3.bias'].numpy()
                                                                       , weights['layer1.0.bn3.running_mean'].numpy()
                                                                       , weights['layer1.0.bn3.running_var'].numpy())
    #(relu): ReLU(inplace=True)
    layer1_bottleneck0_relu = network.add_activation(input=layer1_bottleneck0_bn3.get_output(0), type=trt.ActivationType.RELU)
    #  (downsample): Sequential
    #    (0): AvgPool2d(kernel_size=1, stride=1, padding=0)
    #    (1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck0_downsample_avgpool = network.add_pooling(layer1_bottleneck0_relu.get_output(0), trt.PoolingType.AVERAGE, (1, 1))
    layer1_bottleneck0_downsample_avgpool.stride = (1, 1)
    layer1_bottleneck0_downsample_conv_w = weights['layer1.0.downsample.1.weight'].numpy()
    layer1_bottleneck0_downsample_conv = network.add_convolution(  input=layer1_bottleneck0_bn1.get_output(0), num_output_maps=256,
                                                                   kernel_shape=(1, 1),
                                                                   kernel=layer1_bottleneck0_downsample_conv_w,
    layer1_bottleneck0_downsample_bn = layer_bn(network, layer1_bottleneck0_downsample_conv.get_output(0)
                                                                       , weights['layer1.0.downsample.2.weight'].numpy()
                                                                       , weights['layer1.0.downsample.2.bias'].numpy()
                                                                       , weights['layer1.0.downsample.2.running_mean'].numpy()
                                                                       , weights['layer1.0.downsample.2.running_var'].numpy())

    #layer1-Bottleneck1
    # (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer1_bottleneck1_conv1_w = weights['layer1.1.conv1.weight'].numpy()
    layer1_bottleneck1_conv1_b = np.zeros(64)
    layer1_bottleneck1_conv1 = network.add_convolution(input=layer1_bottleneck0_downsample_bn.get_output(0),
                                                       num_output_maps=64,
                                                       kernel_shape=(1, 1),
                                                       kernel=layer1_bottleneck1_conv1_w,
    #(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck1_bn1 = layer_bn(network, layer1_bottleneck1_conv1.get_output(0), weights['layer1.1.bn1.weight'].numpy()
                                                                                   , weights['layer1.1.bn1.bias'].numpy()
                                                                                   , weights['layer1.1.bn1.running_mean'].numpy()
                                                                                   , weights['layer1.1.bn1.running_var'].numpy())
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    layer1_bottleneck1_conv2_w = weights['layer1.1.conv2.weight'].numpy()
    layer1_bottleneck1_conv2 = network.add_convolution(input=layer1_bottleneck1_bn1.get_output(0),
                                                       num_output_maps=64,
                                                       kernel_shape=(3, 3),
                                                       kernel=layer1_bottleneck1_conv2_w,
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck1_bn2 = layer_bn(network, layer1_bottleneck1_conv2.get_output(0), weights['layer1.1.bn2.weight'].numpy()
                                                                       , weights['layer1.1.bn2.bias'].numpy()
                                                                       , weights['layer1.1.bn2.running_mean'].numpy()
                                                                       , weights['layer1.1.bn2.running_var'].numpy())
    # (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer1_bottleneck1_conv3_w = weights['layer1.1.conv3.weight'].numpy()
    layer1_bottleneck1_conv3 = network.add_convolution(input=layer1_bottleneck1_bn2.get_output(0),
    # (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer1_bottleneck1_bn3 = layer_bn(network, layer1_bottleneck1_conv3.get_output(0), weights['layer1.1.bn3.weight'].numpy()
                                                                         , weights['layer1.1.bn3.bias'].numpy()
                                                                         , weights['layer1.1.bn3.running_mean'].numpy()
                                                                         , weights['layer1.1.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer1_bottleneck1_relu = network.add_activation(input=layer1_bottleneck1_bn3.get_output(0), type=trt.ActivationType.RELU)
    ###########################################################################################################################
    # LAYER2-Bottleneck0
    # (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer_conv(network, input=layer1_bottleneck1_relu.get_output(0), output_size=128, w=weights['layer2.0.conv1.weight'].numpy(), kernel_size=(1, 1), stride=(1, 1) )

    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck0_bn1 = layer_bn(network, layer2_bottleneck0_conv1.get_output(0), weights['layer2.0.bn1.weight'].numpy()
                                                                                     , weights['layer2.0.bn1.bias'].numpy()
                                                                                     , weights['layer2.0.bn1.running_mean'].numpy()
                                                                                     , weights['layer2.0.bn1.running_var'].numpy())
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    layer2_bottleneck0_conv2_w = weights['layer2.0.conv2.weight'].numpy()
    layer2_bottleneck0_conv2 = network.add_convolution(input=layer2_bottleneck0_bn1.get_output(0),
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck0_bn2 = layer_bn(network, layer2_bottleneck0_conv2.get_output(0), weights['layer2.0.bn2.weight'].numpy()
                                                                                     , weights['layer2.0.bn2.bias'].numpy()
                                                                                     , weights['layer2.0.bn2.running_mean'].numpy()
                                                                                     , weights['layer2.0.bn2.running_var'].numpy())
    # (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer2_bottleneck0_conv3_w = weights['layer2.0.conv3.weight'].numpy()
    layer2_bottleneck0_conv3 = network.add_convolution(input=layer2_bottleneck0_bn2.get_output(0),
    # (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck0_bn3 = layer_bn(network, layer2_bottleneck0_conv3.get_output(0), weights['layer2.0.bn3.weight'].numpy()
                                                                                     , weights['layer2.0.bn3.bias'].numpy()
                                                                                     , weights['layer2.0.bn3.running_mean'].numpy()
                                                                                     , weights['layer2.0.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer2_bottleneck0_relu =network.add_activation(input=layer2_bottleneck0_bn3.get_output(0), type=trt.ActivationType.RELU)
    # (downsample): Sequential(
    #  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
    #  (1): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #  (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck0_downsample_avgpool = network.add_pooling(layer2_bottleneck0_relu.get_output(0), trt.PoolingType.AVERAGE, (2, 2))
    layer2_bottleneck0_downsample_avgpool.stride = (2, 2)
    layer2_bottleneck0_downsample_conv_w = weights['layer2.0.downsample.1.weight'].numpy()
    layer2_bottleneck0_downsample_conv = network.add_convolution(input=layer2_bottleneck0_downsample_avgpool.get_output(0),
    layer2_bottleneck0_downsample_bn = layer_bn(network, layer2_bottleneck0_downsample_conv.get_output(0), weights['layer2.0.downsample.2.weight'].numpy()
                                                                                                         , weights['layer2.0.downsample.2.bias'].numpy()
                                                                                                         , weights['layer2.0.downsample.2.running_mean'].numpy()
                                                                                                         , weights['layer2.0.downsample.2.running_var'].numpy())
    # LAYER2-Bottleneck1
    # (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer2_bottleneck1_conv1_w = weights['layer2.1.conv1.weight'].numpy()
    layer2_bottleneck1_conv1 = network.add_convolution( input=layer2_bottleneck0_downsample_bn.get_output(0),
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck1_bn1 = layer_bn(network, layer2_bottleneck1_conv1.get_output(0), weights['layer2.1.bn1.weight'].numpy()
                                                                                     , weights['layer2.1.bn1.bias'].numpy()
                                                                                     , weights['layer2.1.bn1.running_mean'].numpy()
                                                                                     , weights['layer2.1.bn1.running_var'].numpy())
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    layer2_bottleneck1_conv2_w = weights['layer2.1.conv2.weight'].numpy()
    layer2_bottleneck1_conv2 = network.add_convolution( input=layer2_bottleneck1_bn1.get_output(0),
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck1_bn2 = layer_bn(network, layer2_bottleneck1_conv2.get_output(0), weights['layer2.1.bn2.weight'].numpy()
                                                                                     , weights['layer2.1.bn2.bias'].numpy()
                                                                                     , weights['layer2.1.bn2.running_mean'].numpy()
                                                                                     , weights['layer2.1.bn2.running_var'].numpy())
    # (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    layer2_bottleneck1_conv3_w = weights['layer2.1.conv3.weight'].numpy()
    layer2_bottleneck1_conv3 = network.add_convolution( input=layer2_bottleneck1_bn2.get_output(0),
    # (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer2_bottleneck1_bn3 = layer_bn(network, layer2_bottleneck1_conv3.get_output(0), weights['layer2.1.bn3.weight'].numpy()
                                                                                     , weights['layer2.1.bn3.bias'].numpy()
                                                                                     , weights['layer2.1.bn3.running_mean'].numpy()
                                                                                     , weights['layer2.1.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer2_bottleneck1_relu = network.add_activation(input=layer2_bottleneck1_bn3.get_output(0), type=trt.ActivationType.RELU)

    #layer3-Bottleneck0
    # (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck0_conv1_w = weights['layer3.0.conv1.weight'].numpy()
    layer3_bottleneck0_conv1 = network.add_convolution( input=layer2_bottleneck1_relu.get_output(0),
    layer3_bottleneck0_bn1 = layer_bn(network, layer3_bottleneck0_conv1.get_output(0), weights['layer3.0.bn1.weight'].numpy()
                                                                                     , weights['layer3.0.bn1.bias'].numpy()
                                                                                     , weights['layer3.0.bn1.running_mean'].numpy()
                                                                                     , weights['layer3.0.bn1.running_var'].numpy())
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck0_conv2_w = weights['layer3.0.conv2.weight'].numpy()
    layer3_bottleneck0_conv2 = network.add_convolution( input=layer3_bottleneck0_bn1.get_output(0),
    layer3_bottleneck0_bn2 = layer_bn(network, layer3_bottleneck0_conv2.get_output(0), weights['layer3.0.bn2.weight'].numpy()
                                                                                     , weights['layer3.0.bn2.bias'].numpy()
                                                                                     , weights['layer3.0.bn2.running_mean'].numpy()
                                                                                     , weights['layer3.0.bn2.running_var'].numpy())
    # (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck0_conv3_w = weights['layer3.0.conv3.weight'].numpy()
    layer3_bottleneck0_conv3 = network.add_convolution( input=layer3_bottleneck0_bn2.get_output(0),
    layer3_bottleneck0_bn3 = layer_bn(network, layer3_bottleneck0_conv3.get_output(0), weights['layer3.0.bn3.weight'].numpy()
                                                                                     , weights['layer3.0.bn3.bias'].numpy()
                                                                                     , weights['layer3.0.bn3.running_mean'].numpy()
                                                                                     , weights['layer3.0.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer3_bottleneck0_relu =  network.add_activation(input=layer3_bottleneck0_bn3.get_output(0), type=trt.ActivationType.RELU)
    # (downsample): Sequential(
    #  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
    layer3_bottleneck0_downsample_avgpool = network.add_pooling(layer3_bottleneck0_relu.get_output(0), trt.PoolingType.AVERAGE, (1, 1))
    layer3_bottleneck0_downsample_avgpool.stride = (1, 1)
    #  (1): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #  (2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck0_downsample_conv_w = weights['layer3.0.downsample.1.weight'].numpy()
    layer3_bottleneck0_downsample_conv = network.add_convolution( input=layer2_bottleneck0_downsample_avgpool.get_output(0),
    layer3_bottleneck0_downsample_bn = layer_bn(network, layer3_bottleneck0_downsample_conv.get_output(0), weights['layer3.0.downsample.2.weight'].numpy()
                                                                                                         , weights['layer3.0.downsample.2.bias'].numpy()
                                                                                                         , weights['layer3.0.downsample.2.running_mean'].numpy()
                                                                                                         , weights['layer3.0.downsample.2.running_var'].numpy())
    # layer3-Bottleneck1
    # (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 
    layer3_bottleneck1_conv1_w = weights['layer3.1.conv1.weight'].numpy()
    layer3_bottleneck1_conv1 = network.add_convolution( input=layer3_bottleneck0_downsample_bn.get_output(0),
    layer3_bottleneck1_bn1 = layer_bn(network, layer3_bottleneck1_conv1.get_output(0), weights['layer3.1.bn1.weight'].numpy()
                                                                                     , weights['layer3.1.bn1.bias'].numpy()
                                                                                     , weights['layer3.1.bn1.running_mean'].numpy()
                                                                                     , weights['layer3.1.bn1.running_var'].numpy())
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck1_conv2_w = weights['layer3.1.conv2.weight'].numpy()
    layer3_bottleneck1_conv2 = network.add_convolution(input=layer3_bottleneck1_bn1.get_output(0),
    layer3_bottleneck1_bn2 = layer_bn(network, layer3_bottleneck1_conv2.get_output(0), weights['layer3.1.bn2.weight'].numpy()
                                                                                     , weights['layer3.1.bn2.bias'].numpy()
                                                                                     , weights['layer3.1.bn2.running_mean'].numpy()
                                                                                     , weights['layer3.1.bn2.running_var'].numpy())
    # (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer3_bottleneck1_conv3_w = weights['layer3.1.conv3.weight'].numpy()
    layer3_bottleneck1_conv3_b = np.zeros(512)
    layer3_bottleneck1_conv3 = network.add_convolution( input=layer3_bottleneck1_bn2.get_output(0),
                                                        num_output_maps=512,
                                                        kernel_shape=(1, 1),
                                                        kernel=layer3_bottleneck1_conv3_w,
                                                        bias=layer3_bottleneck1_conv3_b)
    layer3_bottleneck1_conv3.stride = (1, 1)
    layer3_bottleneck1_bn3 = layer_bn(network, layer3_bottleneck1_conv3, weights['layer3.1.bn3.weight'].numpy()
                                                                     , weights['layer3.1.bn3.bias'].numpy()
                                                                     , weights['layer3.1.bn3.running_mean'].numpy()
                                                                     , weights['layer3.1.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer3_bottleneck1_relu = network.add_activation(input=layer3_bottleneck1_bn3.get_output(0), type=trt.ActivationType.RELU)
    # (layer4): (0): Bottleneck(    # (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck0_conv1_w = weights['layer4.0.conv1.weight'].numpy()
    layer4_bottleneck0_conv1_b = np.zeros(512)
    layer4_bottleneck0_conv1= network.add_convolution(  input=layer3_bottleneck1_relu.get_output(0),
                                                        num_output_maps=512,
                                                        kernel_shape=(1, 1),
                                                        kernel=layer4_bottleneck0_conv1_w,
                                                        bias=layer4_bottleneck0_conv1_b)
    layer4_bottleneck0_conv1.stride = (1, 1)
    layer4_bottleneck0_bn1 = layer_bn(network, layer4_bottleneck0_conv1, weights['layer4.0.bn1.weight'].numpy()
                                                                     , weights['layer4.0.bn1.bias'].numpy()
                                                                     , weights['layer4.0.bn1.running_mean'].numpy()
                                                                     , weights['layer4.0.bn1.running_var'].numpy())
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck0_conv2_w = weights['layer4.0.conv2.weight'].numpy()
    layer4_bottleneck0_conv2_b = np.zeros(512)
    layer4_bottleneck0_conv2   = network.add_convolution( input=layer4_bottleneck0_bn1.get_output(0),
                                                          num_output_maps=512,
                                                          kernel_shape=(1, 1),
                                                          kernel=layer4_bottleneck0_conv2_w,
                                                          bias=layer4_bottleneck0_conv2_b)
    layer4_bottleneck0_conv2.stride = (1, 1)
    layer4_bottleneck0_bn2 = layer_bn(network, layer4_bottleneck0_conv2, weights['layer4.0.bn2.weight'].numpy()
                                                                     , weights['layer4.0.bn2.bias'].numpy()
                                                                     , weights['layer4.0.bn2.running_mean'].numpy()
                                                                     , weights['layer4.0.bn2.running_var'].numpy())
    # (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck0_conv3_w = weights['layer4.0.conv3.weight'].numpy()
    layer4_bottleneck0_conv3_b = np.zeros(512)
    layer4_bottleneck0_conv3 = network.add_convolution( input=layer4_bottleneck0_bn2.get_output(0),
                                                        num_output_maps=512,
                                                        kernel_shape=(1, 1),
                                                        kernel=layer4_bottleneck0_conv3_w,
                                                        bias=layer4_bottleneck0_conv3_b)
    layer4_bottleneck0_conv3.stride = (1, 1)
    layer4_bottleneck0_bn3 = layer_bn(network, layer4_bottleneck0_conv3, weights['layer4.0.bn3.weight'].numpy()
                                                                         , weights['layer4.0.bn3.bias'].numpy()
                                                                         , weights['layer4.0.bn3.running_mean'].numpy()
                                                                         , weights['layer4.0.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer4_bottleneck0_relu = network.add_activation(input=layer4_bottleneck0_bn3.get_output(0), type=trt.ActivationType.RELU)
    # (downsample): Sequential(
    #  (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
    layer4_bottleneck0_downsample_avgpool = network.add_pooling(layer4_bottleneck0_relu.get_output(0), trt.PoolingType.AVERAGE, (1, 1))
    layer4_bottleneck0_downsample_avgpool.stride = (1, 1)
    #  (1): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #  (2): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck0_downsample_conv_w = weights['layer4.0.downsample.1.weight'].numpy()
    layer4_bottleneck0_downsample_conv_b = np.zeros(512)
    layer4_bottleneck0_downsample_conv = network.add_convolution(   input=layer2_bottleneck0_downsample_avgpool.get_output(0),
                                                                    num_output_maps=512,
                                                                    kernel_shape=(1, 1),
                                                                    kernel=layer4_bottleneck0_downsample_conv_w,
                                                                    bias=layer4_bottleneck0_downsample_conv_b)
    layer4_bottleneck0_downsample_conv.stride = (1, 1)
    layer4_bottleneck0_downsample_bn = layer_bn(network, layer4_bottleneck0_downsample_conv, weights['layer4.0.downsample.2.weight'].numpy()
                                                                                 , weights['layer4.0.downsample.2.bias'].numpy()
                                                                                 , weights['layer4.0.downsample.2.running_mean'].numpy()
                                                                                 , weights['layer4.0.downsample.2.running_var'].numpy())

    # (layer4): (1): Bottleneck(
    # (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck1_conv1_w = weights['layer4.1.conv1.weight'].numpy()
    layer4_bottleneck1_conv1_b = np.zeros(512)
    layer4_bottleneck1_conv1   = network.add_convolution(  input=layer4_bottleneck0_downsample_bn.get_output(0),
                                                            num_output_maps=512,
                                                            kernel_shape=(1, 1),
                                                            kernel=layer4_bottleneck1_conv1_w,
                                                            bias=layer4_bottleneck1_conv1_b)
    layer4_bottleneck1_conv1.stride = (1, 1)
    layer4_bottleneck1_bn1 = layer_bn(network, layer4_bottleneck1_conv1, weights['layer4.1.bn1.weight'].numpy()
                                                                      , weights['layer4.1.bn1.bias'].numpy()
                                                                      , weights['layer4.1.bn1.running_mean'].numpy()
                                                                      , weights['layer4.1.bn1.running_var'].numpy())
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck1_conv2_w = weights['layer4.1.conv2.weight'].numpy()
    layer4_bottleneck1_conv2_b = np.zeros(512)
    layer4_bottleneck1_conv2   = network.add_convolution( input=layer4_bottleneck1_bn1.get_output(0),
                                                            num_output_maps=512,
                                                            kernel_shape=(1, 1),
                                                            kernel=layer4_bottleneck1_conv2_w,
                                                            bias=layer4_bottleneck1_conv2_b)
    layer4_bottleneck1_conv2.stride = (1, 1)
    layer4_bottleneck1_bn2 = layer_bn(network, layer4_bottleneck1_conv2, weights['layer4.1.bn2.weight'].numpy()
                                                                         , weights['layer4.1.bn2.bias'].numpy()
                                                                         , weights['layer4.1.bn2.running_mean'].numpy()
                                                                         , weights['layer4.1.bn2.running_var'].numpy())
    # (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    layer4_bottleneck1_conv3_w = weights['layer4.1.conv3.weight'].numpy()
    layer4_bottleneck1_conv3_b = np.zeros(512)
    layer4_bottleneck1_conv3   = network.add_convolution(  input=layer4_bottleneck1_bn2.get_output(0),
                                                            num_output_maps=512,
                                                            kernel_shape=(1, 1),
                                                            kernel=layer4_bottleneck1_conv3_w,
                                                            bias=layer4_bottleneck1_conv3_b)
    layer4_bottleneck1_conv3.stride = (1, 1)
    layer4_bottleneck1_bn3   = layer_bn(network, layer4_bottleneck1_conv3, weights['layer4.1.bn3.weight'].numpy()
                                                                         , weights['layer4.1.bn3.bias'].numpy()
                                                                         , weights['layer4.1.bn3.running_mean'].numpy()
                                                                         , weights['layer4.1.bn3.running_var'].numpy())
    # (relu): ReLU(inplace=True)
    layer4_bottleneck1_relu=network.add_activation(input=layer4_bottleneck1_bn3.get_output(0), type=trt.ActivationType.RELU)
    #(global_pool): SelectAdaptivePool2d(output_size=1, pool_type=avg)
    layer4_bottleneck1_avgpool = network.add_pooling(layer4_bottleneck1_relu.get_output(0), trt.PoolingType.AVERAGE, (1, 1))
    layer4_bottleneck1_avgpool.stride = (1, 1)
    #(fc): Linear(in_features=2048, out_features=1000, bias=True)
    fc_w = weights['fc.weight'].numpy()
    fc_b = weights['fc.bias'].numpy()
    fc   = network.add_fully_connected(layer4_bottleneck1_avgpool.get_output(0), ModelData.OUTPUT_SIZE, fc_w, fc_b)

    fc.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc.get_output(0))
'''
def populate_network_mnist(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1_w = weights['conv1.weight'].numpy()
    conv1_b = weights['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w,
                                    bias=conv1_b)
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv2_w = weights['conv2.weight'].numpy()
    conv2_b = weights['conv2.bias'].numpy()
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        #populate_network_mnist(network, weights)
        populate_network_resnet26d(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output

def get_weights_resnet(model):
        return model.state_dict()
def main():
    import timm
    m = timm.create_model('resnet26d', pretrained=True)
    m.eval()
    weights = get_weights_resnet(m)
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # New API
        with open("resnet26d.engine", "wb") as f:
            f.write(engine.serialize())

        with engine.create_execution_context() as context:
            '''
            case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))
            '''
            print('end')
def main2():
    _, _ = common.find_sample_data(description="Runs an MNIST network using a PyTorch model", subfolder="mnist")
    # Train the PyTorch model
    mnist_model = model.MnistModel()
    mnist_model.learn()
    weights = mnist_model.get_weights()
    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))

if __name__ == '__main__':
    main()
