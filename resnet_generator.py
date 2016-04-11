#!/usr/bin/env python
"""
Generate the residule learning network.
Author: Yongchan Kwon
Email: ykwon0407@snu.ac.kr

Thanks to Yemin Shi (https://github.com/shiyemin/ResNet-Generator-for-caffe), 
this script also generates ResNet. This version is more likely to MSRA version. 
For resnet-50, the first convolution layer has bias term.

MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf

Disclaimer: This work was highly referred Yemin Shi's script
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

word = ['o','_1','_2','_3']

def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--deploy_file', action='store',
                        default='resnet_tiny_deploy.prototxt')
    parser.add_argument('--layer_number', nargs='*', action='store',type=int,
                        help=('Layer number for each layer stage.'),
                        default=[2, 2, 4, 2])
    args = parser.parse_args()
    print (parser.parse_args())
    return args

def generate_data_layer():
    data_layer_str = '''name: "ResNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 448
input_dim: 448
'''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, filler="xavier"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    bias_term: false
    weight_filler {
      type: "%s"
    }
  }
}
'''%(layer_name, bottom, layer_name, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_bn_layer(batch_name, scale_name, bottom):
    bn_layer_str = '''layer {
	bottom: "%s"
	top: "%s"
	name: "%s"
	type: "BatchNorm"
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 0
	}
	param {
    		lr_mult: 0
  	}
}
layer {
	bottom: "%s"
	top: "%s"
	name: "%s"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}
'''%(bottom, batch_name, batch_name, batch_name, scale_name, scale_name)
    return bn_layer_str
    
def generate_activation_layer(layer_name, bottom):
    act_layer_str = '''layer {
  name: "%s"
  type: "ReLU"
  bottom: "%s"
  top: "%s"
}
'''%(layer_name, bottom, bottom)
    return act_layer_str
    
def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}
'''%(layer_name, bottom, layer_name, pool_type, kernel_size, stride)
    return pool_layer_str
    
def generate_eltwise_layer(layer_name, bottom_1, bottom_2):
    eltwise_layer_str = '''layer {
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  name: "%s"
  top: "%s"
}
'''%(bottom_1, bottom_2, layer_name, layer_name)
    return eltwise_layer_str
    
def generate_fc_layer(num_output, layer_name, bottom):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
      type: "xavier"
    }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}
'''%(layer_name, bottom, layer_name, num_output)
    return fc_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss2/softmax"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss2/softmax"
  include {
    phase: TEST
  }
}
layer {
  bottom: "%s"
  bottom: "label"
  top: "loss1/acc"
  name: "loss1/acc"
  type: "Accuracy"
  include {
    phase: TEST
  }
}
'''%(bottom, bottom)
    return softmax_loss_str

def generate_deploy():
    args = parse_args()
    network_str = generate_data_layer()
    '''before stage'''
    last_top = 'data'
    network_str += generate_conv_layer(3, 32, 2, 1, 'conv1', last_top)
    network_str += generate_bn_layer('bn_conv1', 'scale_conv1', 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'scale_conv1')
    network_str += generate_conv_layer(3, 64, 2, 1, 'new_conv1', 'scale_conv1')
    network_str += generate_bn_layer('bn_new_conv1', 'scale_new_conv1', 'new_conv1')
    network_str += generate_activation_layer('new_conv1_relu', 'scale_new_conv1')
    network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'scale_new_conv1')
    '''stage 1'''
    last_top = 'pool1'
    network_str += generate_conv_layer(1, 128, 1, 0, 'res2_1_branch1', last_top)
    network_str += generate_bn_layer('bn2_1_branch1', 'scale2_1_branch1', 'res2_1_branch1')
    last_output = 'scale2_1_branch1'
    for l in xrange(1, args.layer_number[0]+1):
        network_str += generate_conv_layer(1, 32, 1, 0, 'res2%s_branch2a'%word[l], last_top)
        network_str += generate_bn_layer('bn2%s_branch2a'%word[l], 'scale2%s_branch2a'%word[l], 'res2%s_branch2a'%word[l])
        network_str += generate_activation_layer('res2%s_branch2a_relu'%word[l], 'scale2%s_branch2a'%word[l])
        network_str += generate_conv_layer(3, 32, 1, 1, 'res2%s_branch2b'%word[l], 'scale2%s_branch2a'%word[l])
        network_str += generate_bn_layer('bn2%s_branch2b'%word[l], 'scale2%s_branch2b'%word[l], 'res2%s_branch2b'%word[l])
        network_str += generate_activation_layer('res2%s_branch2b_relu'%word[l], 'scale2%s_branch2b'%word[l])
        network_str += generate_conv_layer(1, 128, 1, 0, 'res2%s_branch2c'%word[l], 'scale2%s_branch2b'%word[l])
        network_str += generate_bn_layer('bn2%s_branch2c'%word[l], 'scale2%s_branch2c'%word[l], 'res2%s_branch2c'%word[l])
        network_str += generate_eltwise_layer('res2%s'%word[l], last_output, 'scale2%s_branch2c'%word[l])
        network_str += generate_activation_layer('res2%s_relu'%word[l],'res2%s'%word[l])
        last_top = 'res2%s'%word[l]
        last_output = 'res2%s'%word[l]
    '''Random Initialization'''
    network_str += generate_conv_layer(1, 256, 2, 0, 'res3_1_branch1', last_top)
    network_str += generate_bn_layer('bn3_1_branch1', 'scale3_1_branch1', 'res3_1_branch1')
    last_output = 'scale3_1_branch1'
    '''stage 2'''
    network_str += generate_conv_layer(1, 64, 2, 0, 'res3_1_branch2a', last_top)
    network_str += generate_bn_layer('bn3_1_branch2a', 'scale3_1_branch2a', 'res3_1_branch2a')
    network_str += generate_activation_layer('res3_1_branch2a_relu', 'scale3_1_branch2a')
    network_str += generate_conv_layer(3, 64, 1, 1, 'res3_1_branch2b', 'scale3_1_branch2a')
    network_str += generate_bn_layer('bn3_1_branch2b', 'scale3_1_branch2b', 'res3_1_branch2b')
    network_str += generate_activation_layer('res3_1_branch2b_relu', 'scale3_1_branch2b')
    network_str += generate_conv_layer(1, 256, 1, 0, 'res3_1_branch2c', 'scale3_1_branch2b')
    network_str += generate_bn_layer('bn3_1_branch2c', 'scale3_1_branch2c', 'res3_1_branch2c')
    network_str += generate_eltwise_layer('res3_1', last_output, 'scale3_1_branch2c')
    network_str += generate_activation_layer('res3_1_relu', 'res3_1')
    last_top = 'res3_1'
    for l in xrange(2, args.layer_number[1]+1):
        network_str += generate_conv_layer(1, 64, 1, 0, 'res3_%d_branch2a'%l, last_top)
        network_str += generate_bn_layer('bn3_%d_branch2a'%l, 'scale3_%d_branch2a'%l, 'res3_%d_branch2a'%l)
        network_str += generate_activation_layer('res3_%d_branch2a_relu'%l, 'scale3_%d_branch2a'%l)
        network_str += generate_conv_layer(3, 64, 1, 1, 'res3_%d_branch2b'%l, 'scale3_%d_branch2a'%l)
        network_str += generate_bn_layer('bn3_%d_branch2b'%l, 'scale3_%d_branch2b'%l, 'res3_%d_branch2b'%l)
        network_str += generate_activation_layer('res3_%d_branch2b_relu'%l, 'scale3_%d_branch2b'%l)
        network_str += generate_conv_layer(1, 256, 1, 0, 'res3_%d_branch2c'%l, 'scale3_%d_branch2b'%l)
        network_str += generate_bn_layer('bn3_%d_branch2c'%l, 'scale3_%d_branch2c'%l, 'res3_%d_branch2c'%l)
        network_str += generate_eltwise_layer('res3_%d'%l, last_top, 'scale3_%d_branch2c'%l)
        network_str += generate_activation_layer('res3_%d_relu'%l, 'res3_%d'%l)
        last_top = 'res3_%d'%l
    network_str += generate_conv_layer(1, 512, 2, 0, 'res4_1_branch1', last_top)
    network_str += generate_bn_layer('bn4_1_branch1', 'scale4_1_branch1', 'res4_1_branch1')
    last_output = 'scale4_1_branch1'
    '''stage 3'''
    network_str += generate_conv_layer(1, 128, 2, 0, 'res4_1_branch2a', last_top)
    network_str += generate_bn_layer('bn4_1_branch2a', 'scale4_1_branch2a', 'res4_1_branch2a')
    network_str += generate_activation_layer('res4_1_branch2a_relu', 'scale4_1_branch2a')
    network_str += generate_conv_layer(3, 128, 1, 1, 'res4_1_branch2b', 'scale4_1_branch2a')
    network_str += generate_bn_layer('bn4_1_branch2b', 'scale4_1_branch2b', 'res4_1_branch2b')
    network_str += generate_activation_layer('res4_1_branch2b_relu', 'scale4_1_branch2b')
    network_str += generate_conv_layer(1, 512, 1, 0, 'res4_1_branch2c', 'scale4_1_branch2b')
    network_str += generate_bn_layer('bn4_1_branch2c', 'scale4_1_branch2c', 'res4_1_branch2c')
    network_str += generate_eltwise_layer('res4_1', last_output, 'scale4_1_branch2c')
    network_str += generate_activation_layer('res4_1_relu', 'res4_1')
    last_top = 'res4_1'
    for l in xrange(2, args.layer_number[2]+1):
    	network_str += generate_conv_layer(1, 128, 1, 0, 'res4_%d_branch2a'%l, last_top)
        network_str += generate_bn_layer('bn4_%d_branch2a'%l, 'scale4_%d_branch2a'%l, 'res4_%d_branch2a'%l)
        network_str += generate_activation_layer('res4_%d_branch2a_relu'%l, 'scale4_%d_branch2a'%l)
        network_str += generate_conv_layer(3, 128, 1, 1, 'res4_%d_branch2b'%l, 'scale4_%d_branch2a'%l)
        network_str += generate_bn_layer('bn4_%d_branch2b'%l, 'scale4_%d_branch2b'%l, 'res4_%d_branch2b'%l)
        network_str += generate_activation_layer('res4_%d_branch2b_relu'%l, 'scale4_%d_branch2b'%l)
        network_str += generate_conv_layer(1, 512, 1, 0, 'res4_%d_branch2c'%l, 'scale4_%d_branch2b'%l)
        network_str += generate_bn_layer('bn4_%d_branch2c'%l, 'scale4_%d_branch2c'%l, 'res4_%d_branch2c'%l)
        network_str += generate_eltwise_layer('res4_%d'%l, last_top, 'scale4_%d_branch2c'%l)
        network_str += generate_activation_layer('res4_%d_relu'%l, 'res4_%d'%l)
        last_top = 'res4_%d'%l
    network_str += generate_conv_layer(1, 1024, 2, 0, 'res5_1_branch1', last_top)
    network_str += generate_bn_layer('bn5_1_branch1', 'scale5_1_branch1', 'res5_1_branch1')
    last_output = 'scale5_1_branch1'
    '''stage 4'''
    network_str += generate_conv_layer(1, 256, 2, 0, 'res5_1_branch2a', last_top)
    network_str += generate_bn_layer('bn5_1_branch2a', 'scale5_1_branch2a', 'res5_1_branch2a')
    network_str += generate_activation_layer('res5_1_branch2a_relu', 'scale5_1_branch2a')
    network_str += generate_conv_layer(3, 256, 1, 1, 'res5_1_branch2b', 'scale5_1_branch2a')
    network_str += generate_bn_layer('bn5_1_branch2b', 'scale5_1_branch2b', 'res5_1_branch2b')
    network_str += generate_activation_layer('res5_1_branch2b_relu', 'scale5_1_branch2b')
    network_str += generate_conv_layer(1, 1024, 1, 0, 'res5_1_branch2c', 'scale5_1_branch2b')
    network_str += generate_bn_layer('bn5_1_branch2c', 'scale5_1_branch2c', 'res5_1_branch2c')
    network_str += generate_eltwise_layer('res5_1', last_output, 'scale5_1_branch2c')
    network_str += generate_activation_layer('res5_1_relu', 'res5_1')
    last_top = 'res5_1'
    for l in xrange(2, args.layer_number[3]+1):
    	network_str += generate_conv_layer(1, 256, 1, 0, 'res5_%d_branch2a'%l, last_top)
        network_str += generate_bn_layer('bn5_%d_branch2a'%l, 'scale5_%d_branch2a'%l, 'res5_%d_branch2a'%l)
        network_str += generate_activation_layer('res5_%d_branch2a_relu'%l, 'scale5_%d_branch2a'%l)
        network_str += generate_conv_layer(3, 256, 1, 1, 'res5_%d_branch2b'%l, 'scale5_%d_branch2a'%l)
        network_str += generate_bn_layer('bn5_%d_branch2b'%l, 'scale5_%d_branch2b'%l, 'res5_%d_branch2b'%l)
        network_str += generate_activation_layer('res5_%d_branch2b_relu'%l, 'scale5_%d_branch2b'%l)
        network_str += generate_conv_layer(1, 1024, 1, 0, 'res5_%d_branch2c'%l, 'scale5_%d_branch2b'%l)
        network_str += generate_bn_layer('bn5_%d_branch2c'%l, 'scale5_%d_branch2c'%l, 'res5_%d_branch2c'%l)
        network_str += generate_eltwise_layer('res5_%d'%l, last_top, 'scale5_%d_branch2c'%l)
        network_str += generate_activation_layer('res5_%d_relu'%l, 'res5_%d'%l)
        last_top = 'res5_%d'%l
    network_str += generate_pooling_layer(7, 1, 'AVE', 'pool5', last_top)
    network_str += generate_fc_layer(5, 'fc5', 'pool5')
    network_str += generate_softmax_loss('fc5')
    return network_str

def main():
    args = parse_args()
    network_str = generate_deploy()
    fp = open(args.deploy_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
