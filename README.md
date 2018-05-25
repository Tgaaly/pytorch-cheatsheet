# New in Pytorch v0.4.0

- `Tensors` and `Variable` have merged
- Deprecation of `volatile`

More information here: https://github.com/pytorch/pytorch/releases

# pytorch-cheatsheet

A cheatsheet to cover the most commonly used aspects of PyTorch!

Pytorch Documentation: 
http://pytorch.org/docs/master/

Pytorch Forums: 
https://discuss.pytorch.org/

## Tensor Types and Operations

##### Commonly used types of tensors
`torch.IntTensor`

`torch.FloatTensor`

`torch.DoubleTensor`

`torch.LongTensor`

##### Common pytorch functions
```
torch.sigmoid(x)
torch.log(x)
torch.sum(x, dim=<dim>)
torch.div(x, y)
```
and many others: `neg(), reciprocal(), pow(), sin(), tanh(), sqrt(), sign()`

##### Convert numpy array to pytorch tensor
`b = torch.from_numpy(a)`

The above keeps the original dtype of the data (e.g. float64 becomes `torch.DoubleTensor`). To override this you can do the following (to cast it to a `torch.FloatTensor`):

`b = torch.FloatTensor(a)`

##### Moving data from CPU to GPU
`data = data.cuda()`

##### Pytorch Variable
The pytorch `torch.autograd.Variable` has a "data" tensor under it:
`data_variable.data`

##### Moving data from GPU to CPU
`data = data.cpu()`

##### Convert tensor to numpy array
`data_arr = data_tensor.numpy()`

##### Moving a Variable to CPU and converting to numpy array
`data = data.data.cpu().numpy()`

##### Viewing the size of a tensor
`data.size()`

##### Add a dimenstion to a torch tensor
`<tensor>.unsqueeze(axis)`

##### Reshaping tensors
The equivalent of numpy's `reshape()` in torch is `view()`

Examples:

```
a = torch.range(1, 16)
a = a.view(4, 4)        # reshapes from 1 x 16 to 4 x 4
```

`<tensor>.view(-1)` vectorizes a tensor.

##### Operations between Pytorch Variables and Numpy variables
The Numpy variables have to be first converted to `torch.Tensor` and then converted to pytorch `torch.autograd.Variable`. 
An example is shown below.

```
matrix_tensor = torch.Tensor(matrix_numpy)
# Use cuda() if everything will be calculated on GPU
matrix_pytorch_variable_cuda_from_numpy = torch.autograd.Variable(matrix_tensor, requires_grad=False).cuda()
loss = F.mse_loss(matrix_pytorch_variable_cuda, matrix_pytorch_variable_cuda_from_numpy)
```

##### Batch matrix operations
```
# Batch matrix multiply
torch.btorch.bmm(batch1, batch2, out=None)
```
##### Transpose a tensor
Transpose axis1 and axis2
`<tensor>.transpose(axis1, axis2)`

##### Outer product
Outer product between two vectors `vec1` and `vec2`
```
output = vec1.unsqueeze(2)*vec2.unsqueeze(1)
output = output.view(output.size(0),-1)
```

## Running on multiple GPUs

##### Multiple GPUs/CPUs for training

Instantiate the model first and then call DataParallel. todo: Add a way to specify the number of GPUs.

`model = Net()`

`model = torch.nn.DataParallel(model)`

For specifying the GPU devices: 
`model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()`

Pytorch 0.2.0 supports distributed data parallelism, i.e. training over multiple nodes (CPUs and GPUs)

##### Setting the GPUs

Usage of the `torch.cuda.set_device(gpu_idx)` is discouraged in favor of `device()`. In most cases it’s better to use the `CUDA_VISIBLE_DEVICES` environmental variable.



## Datasets and Data Loaders

##### Creating a dataset and enumerating over it
Inherit from `torch.utils.data.Dataset` and overload `__getitem__()` and `__len()__`

Example:

```
class FooDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    ...
  def __getitem__(self, idx):
    ...
    return {'data': batch_data, 'label': batch_label}
  def __len__(self):
    ...
    return <length of dataset>
```

To loop over a datasets batches:
```
foo_dataset = FooDataset(root_dir)
data_loader = torch.utils.data.DataLoader(foo_dataset, batch_size=<batch_size>, shuffle=True)

for batch_idx, batch in enumerate(data_loader):
  data = batch['data']
  label = batch['label']
  
  if args.cuda:
    data, label = data.cuda(), label.cuda()
    
  data = Variable(data)
  label = Variable(label)
```

## Setting Torch Random Seed
Use `torch.manual_seed(seed)` in addition to `np.random.seed(seed)` to make training deterministic. I believe in the future torch will use the numpy seed so they won't be separate anymore. 

## Convolutional Layers

##### Conv2d layer
`torch.nn.Conv2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), bias=False, dilation=<d>)`


##### Transpose Conv2d layer ('Deconvolution layer')
`torch.nn.ConvTranspose2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), output_padding=(x,y), bias=False, dilation=<d>)`


## Model Inference

##### Volatile at inference time
Don't forget to set the input to the graph/net to `volatile=True`. Even if you do `model.eval()`, if the input data is not set to volatile then memory will be used up to compute the gradients. `model.eval()` sets batchnorm and dropout to inference/test mode, insteasd of training mode which is the default when the model is instantiated. If at least one torch Variable is not volatile in the graph (including the input variable being fed into the network graph), it will cause gradients to be computed in the graph even if `model.eval()` was called. This will take up extra memory. 

Important: `volatile` has been deprecated as of v0.4.0. Now it has been replaced with `requires_grad` (attribute of `Tensor`), `torch.no_grad()`, `torch.set_grad_enabled(grad_mode)`. See information here: https://github.com/pytorch/pytorch/releases

Example:

`data = Variable(data, volatile=True)`

`output = model(data)`

##### Deploying/Serving Pytorch to Production Using TensorRT

[copied from here https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html]

Using NVIDIA TensorRT. NVIDIA TensorRT is a C++ library that facilitates high performance inference on NVIDIA graphics processing units (GPUs). TensorRT takes a network definition and optimizes it by merging tensors and layers, transforming weights, choosing efficient intermediate data formats, and selecting from a large kernel catalog based on layer parameters and measured performance.

TensorRT consists of import methods to help you express your trained deep learning model for TensorRT to optimize and run. It is an optimization tool that applies graph optimization and layer fusion and finds the fastest implementation of that model leveraging a diverse collection of highly optimized kernels, and a runtime that you can use to execute this network in an inference context.

TensorRT includes an infrastructure that allows you to leverage high speed reduced precision capabilities of Pascal GPUs as an optional optimization.

For installation and setup, see link above. I would recommend following the tar installation. I found an error with `cuda.h` not being found so had to make sure my cuda version was properly setup and upgraded to cuda-9.0. The tar installation should lead you through installing pycuda, tensorRT and UFF.

A summary of the steps I did to get this work was:
* Install pycuda first: `pip install 'pycuda>=2017.1.1'` (had problems with pycuda installation. couldnt find cuda.h - so installed cuda-9.0 and updated `PATH` and `LD_LIBRARY_PATH` in `~/.bashrc` and sourced.
* Downloaded tensorRT tar and followed instructions to install (e.g. `pip install tensorRT/python/<path-to-wheel>.whl` and `pip install tensorRT/uff/<path-to-wheel>.whl`).
* To verify I made sure I could do the following (of course you have to install tensorflow - see below):
```
    import tensorflow
    import uff 
    import tensorrt as trr
```
This worked with tensorRT v4.0.0.3, cuda-9.0, tensorflow version: 1.4.1, pytorch version: 0.3.0.post4. Pytorch was needed for the example below of converting a pytorch model to run on an tensorRT engine.

Once installed you have to also install tensorflow.
For tensorflow-gpu with cuda8 use (tensorflow version 1.5 uses cuda 9.0): `pip install tensorflow-gpu==1.4.1`
else just use `pip install tensorflow-gpu` for the latest version.

Example of doing this for Pytorch and tensorRT 3.0 (this also worked for my tensorRT version of 4.0): 
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/topics/topics/workflows/manually_construct_tensorrt_engine.html

## Portability to Other Frameworks

Pytorch now (as of v0.3.0) supports exporting models to other frameworks starting with Caffe2AI and MSCNTK. So now models can be deployed/served!

<!--Pytorch v0.3.0: Model Exporter to ONNX (ship PyTorch to Caffe2 (part of Pytorch now), CoreML, CNTK, MXNet, Tensorflow)-->

## Saving/Loading Pytorch Models

<!--
It is not recommended to save the entire model (architecture and weights) the way that you did because that method will not work if you try to load the model in a different project. For example, if you try to send your model file ./torch_model_v1 to me and I try to load it with torch.load("./torch_model_v1") I will get an error because it’s likely my project won’t have the exact same directory structure as your project.
Instead you should save only the model weights (state dict), define the architecture in code, then load the weights into the new models state dict.
-->

<!--Currently there is no supported way within Pytorch to serve/deploy models efficiently. -->

For the sake of resuming training Pytorch allows saving and loading the models via two means - see http://pytorch.org/docs/master/notes/serialization.html. 
Beware that load/save pytorch models breaks down if the directory structure or class definitions change so when its time to deploy the model (and by this I mean running it purely from python on another machine for example) the model class has to be added to the python path in order for the class to be instantiated. It's actually very weird. If you save a model, change the directory structure (e.g. put the model in a subfolder) and try to load the model - it will not load. It will complain that it cannot find the class definition. The work around would be to add the class definition to your python path. This is written as a note on the Pytorch documentation page http://pytorch.org/docs/master/notes/serialization.html. See example here:

Save the models weights and define the model architecture in the code. You can then load the weights into the new model state dict. 

### Save 
```
torch.save(model.state_dict(), "./torch_model_v1.pt")
```

### Load
```
model = Model() # the model should be defined with the same code you used to create the trained model
state_dict = torch.load( "./torch_model_v1.pt")
model.load_state_dict(state_dict)
```

[taken from https://discuss.pytorch.org/t/using-a-pytorch-model-for-inference/14770/2]

## Loss Functions

A list of all the ready-made losses is here: http://pytorch.org/docs/master/nn.html#loss-functions

In Pytorch you can write any loss you want as long as you stick to using Pytorch `Variables` (without any `.data` unpacking or numpy conversions) and `torch` functions. The loss will not backprop (when using `loss.backward()`) if you use numpy data structures.

## Training an RNN with features from a CNN
Use `torch.stack(feature_seq, dim=1)` to stack all the features from the CNNs into a sequence. Then feed this into the RNN. Remember you can specify the batch size as the first dimension of the input tensor but you have to set the `batch_first=True` argument when instantiating the RNN (by default it is set to False).

Example:
```
 self.rnn1 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
```




