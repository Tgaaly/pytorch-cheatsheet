# pytorch-cheatsheet


##### Add a dimenstion to a torch tensor
`<tensor>.unsqueeze(axis)`

##### Running on multiple GPUs
After instantiating the model (model = Net())
`model = torch.nn.DataParallel(model)`

##### Vectorize a tensor
`<tensor>.view(-1)`

##### Volatile at inference time
Don't forget to set the input to the graph/net to `volatile=True`

Example:
`data = Variable(data, volatile=True)`
`output = model(data)`

#####

#####

#####

#####
