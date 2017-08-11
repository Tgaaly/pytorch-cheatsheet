# pytorch-cheatsheet


##### Commonly used types of tensors
`torch.FloatTensor`

`torch.DoubleTensor`

`torch.LongTensor`

##### Add a dimenstion to a torch tensor
`<tensor>.unsqueeze(axis)`

##### Running on multiple GPUs
Instantiate the model first and then call DataParallel. todo: Add a way to specify the number of GPUs.

`model = Net()`

`model = torch.nn.DataParallel(model)`

##### Vectorize a tensor
`<tensor>.view(-1)`

##### Volatile at inference time
Don't forget to set the input to the graph/net to `volatile=True`. Even if you do `model.eval()`, if the input data is not set to volatile then memory will be used up to compute the gradients.

Example:

`data = Variable(data, volatile=True)`

`output = model(data)`

##### Transpose a tensor
Transpose axis1 and axis2
`<tensor>.transpose(axis1, axis2)`

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

##### Conv2d layer
`torch.nn.Conv2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), bias=False, dilation=<d>)`


##### Transpose Conv2d layer ('Deconvolution layer')
`torch.nn.ConvTranspose2d(in_channels, out_channels, (kernel_w, kernel_h), stride=(x,y), padding=(x,y), output_padding=(x,y), bias=False, dilation=<d>)`
