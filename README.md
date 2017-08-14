# pytorch-cheatsheet

## Tensor Types and Operations

##### Commonly used types of tensors
`torch.FloatTensor`

`torch.DoubleTensor`

`torch.LongTensor`

##### Common pytorch functions
```
torch.log(x)
torch.sum(x, dim=<dim>)
torch.div(x, y)
```
##### Convert numpy array to pytorch tensor
`b = torch.from_numpy(a)`

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

##### Running on multiple GPUs
Instantiate the model first and then call DataParallel. todo: Add a way to specify the number of GPUs.

`model = Net()`

`model = torch.nn.DataParallel(model)`

##### Setting the GPUs

Usage of the `torch.cuda.set_device(gpu_idx)` is discouraged in favor of `device()`. In most cases it’s better to use the `CUDA_VISIBLE_DEVICES` environmental variable.

##### Vectorize a tensor
`<tensor>.view(-1)`

##### Transpose a tensor
Transpose axis1 and axis2
`<tensor>.transpose(axis1, axis2)`

## Datasets and Data Loaders

##### Creating a dataset and enumerating over it
Inherit from torch.utils.data and overload `__getitem__()` and `__len()__`

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

## Convolutional Layers

##### Conv2d layer
`torch.nn.Conv2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), bias=False, dilation=<d>)`


##### Transpose Conv2d layer ('Deconvolution layer')
`torch.nn.ConvTranspose2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), output_padding=(x,y), bias=False, dilation=<d>)`


## Model Inference

##### Volatile at inference time
Don't forget to set the input to the graph/net to `volatile=True`. Even if you do `model.eval()`, if the input data is not set to volatile then memory will be used up to compute the gradients.

Example:

`data = Variable(data, volatile=True)`

`output = model(data)`
