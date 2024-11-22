# PyTorch Reference

## Core Libraries and Modules

### torch
* Tensor Operations
  * Creation
    * `torch.tensor(data, dtype=None, device=None)`
    * `torch.zeros(size, dtype=None, device=None)`
    * `torch.ones(size, dtype=None, device=None)`
    * `torch.rand(size, dtype=None, device=None)`
    * `torch.randn(size, dtype=None, device=None)`
    * `torch.arange(start, end, step=1, dtype=None)`
    * `torch.linspace(start, end, steps=100)`

  * Manipulation
    * `torch.cat(tensors, dim=0)`
    * `torch.stack(tensors, dim=0)`
    * `torch.chunk(input, chunks, dim=0)`
    * `torch.split(tensor, split_size_or_sections, dim=0)`
    * `torch.reshape(input, shape)`
    * `torch.transpose(input, dim0, dim1)`
    * `torch.squeeze(input, dim=None)`
    * `torch.unsqueeze(input, dim)`

  * Math Operations
    * `torch.add(input, other)`
    * `torch.sub(input, other)`
    * `torch.mul(input, other)`
    * `torch.div(input, other)`
    * `torch.mm(input, mat2)`
    * `torch.matmul(input, other)`
    * `torch.exp(input)`
    * `torch.log(input)`
    * `torch.pow(input, exponent)`
    * `torch.sqrt(input)`

### torch.nn
* Layers
  * Linear Layers
    * `nn.Linear(in_features, out_features, bias=True)`
    * `nn.LazyLinear(out_features)`
    * `nn.Bilinear(in1_features, in2_features, out_features)`

  * Convolutional Layers
    * `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
    * `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
    * `nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
    * `nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`

  * Recurrent Layers
    * `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh')`
    * `nn.LSTM(input_size, hidden_size, num_layers=1, bias=True)`
    * `nn.GRU(input_size, hidden_size, num_layers=1, bias=True)`

  * Transformer Layers
    * `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)`
    * `nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)`
    * `nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)`

  * Normalization Layers
    * `nn.BatchNorm1d(num_features)`
    * `nn.BatchNorm2d(num_features)`
    * `nn.LayerNorm(normalized_shape)`
    * `nn.InstanceNorm2d(num_features)`

  * Pooling Layers
    * `nn.MaxPool1d(kernel_size, stride=None, padding=0)`
    * `nn.MaxPool2d(kernel_size, stride=None, padding=0)`
    * `nn.AvgPool2d(kernel_size, stride=None, padding=0)`
    * `nn.AdaptiveAvgPool2d(output_size)`

  * Activation Functions
    * `nn.ReLU(inplace=False)`
    * `nn.LeakyReLU(negative_slope=0.01)`
    * `nn.Sigmoid()`
    * `nn.Tanh()`
    * `nn.GELU()`
    * `nn.Softmax(dim=None)`

  * Dropout Layers
    * `nn.Dropout(p=0.5, inplace=False)`
    * `nn.Dropout2d(p=0.5, inplace=False)`
    * `nn.AlphaDropout(p=0.5)`

### torch.nn.functional
* Activation Functions
  * `F.relu(input, inplace=False)`
  * `F.leaky_relu(input, negative_slope=0.01)`
  * `F.sigmoid(input)`
  * `F.tanh(input)`
  * `F.softmax(input, dim=None)`
  * `F.gelu(input)`

* Loss Functions
  * `F.binary_cross_entropy(input, target)`
  * `F.cross_entropy(input, target)`
  * `F.mse_loss(input, target)`
  * `F.l1_loss(input, target)`
  * `F.nll_loss(input, target)`
  * `F.kl_div(input, target)`

* Other Functions
  * `F.pad(input, pad, mode='constant', value=0)`
  * `F.interpolate(input, size=None, scale_factor=None, mode='nearest')`
  * `F.one_hot(tensor, num_classes=-1)`
  * `F.embedding(input, weight, padding_idx=None)`

### torch.optim
* Optimizers
  * `optim.SGD(params, lr=0.01, momentum=0, dampening=0)`
  * `optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)`
  * `optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)`
  * `optim.RMSprop(params, lr=0.01, alpha=0.99)`
  * `optim.Adagrad(params, lr=0.01, lr_decay=0)`
  * `optim.Adadelta(params, lr=1.0, rho=0.9)`

* Learning Rate Schedulers
  * `optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)`
  * `optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)`
  * `optim.lr_scheduler.ExponentialLR(optimizer, gamma)`
  * `optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)`
  * `optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')`

### torch.utils.data
* Data Loading
  * `DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)`
  * `Dataset`
  * `TensorDataset`
  * `random_split(dataset, lengths)`
  * `Subset(dataset, indices)`
  * `ConcatDataset(datasets)`

### torchvision
* Transforms
  * Image Transforms
    * `transforms.Resize(size)`
    * `transforms.RandomResizedCrop(size)`
    * `transforms.RandomHorizontalFlip(p=0.5)`
    * `transforms.RandomVerticalFlip(p=0.5)`
    * `transforms.RandomRotation(degrees)`
    * `transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`
    * `transforms.Normalize(mean, std)`
    * `transforms.ToTensor()`
    * `transforms.ToPILImage()`

  * Transform Compositions
    * `transforms.Compose(transforms)`

* Models
  * Pre-trained Models
    * `models.resnet18(pretrained=False)`
    * `models.resnet50(pretrained=False)`
    * `models.vgg16(pretrained=False)`
    * `models.densenet121(pretrained=False)`
    * `models.efficientnet_b0(pretrained=False)`
    * `models.mobilenet_v2(pretrained=False)`

### torch.cuda
* GPU Operations
  * `torch.cuda.is_available()`
  * `torch.cuda.device_count()`
  * `torch.cuda.current_device()`
  * `torch.cuda.get_device_name(device)`
  * `torch.cuda.empty_cache()`
  * `torch.cuda.synchronize()`

### torch.distributed
* Distributed Training
  * `init_process_group(backend, init_method=None, rank=None, world_size=None)`
  * `DistributedDataParallel(module, device_ids=None)`
  * `all_reduce(tensor, op=ReduceOp.SUM)`
  * `broadcast(tensor, src)`
  * `barrier()`

### torch.jit (TorchScript)
* Script Compilation
  * `torch.jit.script(obj)`
  * `torch.jit.trace(func, example_inputs)`
  * `torch.jit.save(m, f)`
  * `torch.jit.load(f)`

## Model Training Utilities

### Training Loop Essentials
* Model Methods
  * `model.train()`
  * `model.eval()`
  * `model.to(device)`
  * `model.parameters()`
  * `model.state_dict()`
  * `model.load_state_dict(state_dict)`

* Gradient Operations
  * `loss.backward()`
  * `optimizer.zero_grad()`
  * `optimizer.step()`
  * `torch.autograd.grad(outputs, inputs)`
  * `torch.no_grad()`

### Model Saving/Loading
* `torch.save(obj, f)`
* `torch.load(f, map_location=None)`

### Metrics and Utilities
* `torchmetrics.Accuracy()`
* `torchmetrics.Precision()`
* `torchmetrics.Recall()`
* `torchmetrics.F1Score()`
* `torchmetrics.AUROC()`