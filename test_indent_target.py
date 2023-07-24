def load(path, **configs):
    """
    :api_attr: imperative

    Load model saved by ``paddle.jit.save`` or ``paddle.static.save_inference_model`` or
    paddle 1.x API ``paddle.fluid.io.save_inference_model`` as ``paddle.jit.TranslatedLayer``,
    then performing inference or fine-tune training.

        .. code-block:: python

            >>> import paddle
            >>> for i in range(3):
            ...     print(i)

    .. note::
        If you load model saved by ``paddle.static.save_inference_model`` ,
        there will be the following limitations when using it in fine-tuning:
        1. Imperative mode do not support LoDTensor. All original model's feed targets or parametars that depend on LoD are temporarily unavailable.
        2. All saved model's feed targets need to be passed into TranslatedLayer's forward function.
        3. The variable's ``stop_gradient`` information is lost and can not be recovered.
        4. The parameter's ``trainable`` information is lost and can not be recovered.

    Args:
        path (str): The path prefix to load model. The format is ``dirname/file_prefix`` or ``file_prefix`` .
        **configs (dict, optional): Other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.


    Returns:
        TranslatedLayer: A Layer object can run saved translated model.

    Examples:
        1. Load model saved by ``paddle.jit.save`` then performing inference and fine-tune training.

        .. code-block:: python
            :name: code-example1

            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt

            >>> BATCH_SIZE = 16
            >>> BATCH_NUM = 4
            >>> EPOCH_NUM = 4

            >>> IMAGE_SIZE = 784
            >>> CLASS_NUM = 10

            >>> # define a random dataset
            >>> class RandomDataset(paddle.io.Dataset):
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples

            ...     def __getitem__(self, idx):
            ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
            ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            ...         return image, label

            ...     def __len__(self):
            ...         return self.num_samples

            >>> class LinearNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

            ...     @paddle.jit.to_static
            ...     def forward(self, x):
            ...         return self._linear(x)

            >>> def train(layer, loader, loss_fn, opt):
            ...     for epoch_id in range(EPOCH_NUM):
            ...         for batch_id, (image, label) in enumerate(loader()):
            ...             out = layer(image)
            ...             loss = loss_fn(out, label)
            ...             loss.backward()
            ...             opt.step()
            ...             opt.clear_grad()
            ...             print("Epoch {} batch {}: loss = {}".format(
            ...                 epoch_id, batch_id, np.mean(loss.numpy())))

            >>> # 1. train & save model.

            >>> # create network
            >>> layer = LinearNet()
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            >>> # create data loader
            >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     num_workers=2)

            >>> # train
            >>> train(layer, loader, loss_fn, adam)

            >>> # save
            >>> path = "example_model/linear"
            >>> paddle.jit.save(layer, path)

            >>> # 2. load model

            >>> # load
            >>> loaded_layer = paddle.jit.load(path)

            >>> # inference
            >>> loaded_layer.eval()
            >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
            >>> pred = loaded_layer(x)

            >>> # fine-tune
            >>> loaded_layer.train()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
            >>> train(loaded_layer, loader, loss_fn, adam)


        2. Load model saved by ``paddle.fluid.io.save_inference_model`` then performing and fine-tune training.

        .. code-block:: python
            :name: code-example2

            >>> import numpy as np
            >>> import paddle
            >>> import paddle.static as static
            >>> import paddle.nn as nn
            >>> import paddle.optimizer as opt
            >>> import paddle.nn.functional as F

            >>> BATCH_SIZE = 16
            >>> BATCH_NUM = 4
            >>> EPOCH_NUM = 4

            >>> IMAGE_SIZE = 784
            >>> CLASS_NUM = 10

            >>> # define a random dataset
            >>> class RandomDataset(paddle.io.Dataset):
            ...     def __init__(self, num_samples):
            ...         self.num_samples = num_samples

            ...     def __getitem__(self, idx):
            ...         image = np.random.random([IMAGE_SIZE]).astype('float32')
            ...         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            ...         return image, label

            ...     def __len__(self):
            ...         return self.num_samples

            >>> paddle.enable_static()

            >>> image = static.data(name='image', shape=[None, 784], dtype='float32')
            >>> label = static.data(name='label', shape=[None, 1], dtype='int64')
            >>> pred = static.nn.fc(x=image, size=10, activation='softmax')
            >>> loss = F.cross_entropy(input=pred, label=label)
            >>> avg_loss = paddle.mean(loss)

            >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            >>> optimizer.minimize(avg_loss)

            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())

            >>> # create data loader
            >>> dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset,
            ...     feed_list=[image, label],
            ...     places=place,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     return_list=False,
            ...     num_workers=2)

            >>> # 1. train and save inference model
            >>> for data in loader():
            ...     exe.run(
            ...         static.default_main_program(),
            ...         feed=data,
            ...         fetch_list=[avg_loss])

            >>> model_path = "fc.example.model"
            >>> paddle.fluid.io.save_inference_model(
            ...     model_path, ["image"], [pred], exe)

            >>> # 2. load model

            >>> # enable dygraph mode
            >>> paddle.disable_static(place)

            >>> # load
            >>> fc = paddle.jit.load(model_path)

            >>> # inference
            >>> fc.eval()
            >>> x = paddle.randn([1, IMAGE_SIZE], 'float32')
            >>> pred = fc(x)

            >>> # fine-tune
            >>> fc.train()
            >>> loss_fn = nn.CrossEntropyLoss()
            >>> adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
            >>> loader = paddle.io.DataLoader(dataset,
            ...     places=place,
            ...     batch_size=BATCH_SIZE,
            ...     shuffle=True,
            ...     drop_last=True,
            ...     num_workers=2)
            >>> for epoch_id in range(EPOCH_NUM):
            ...     for batch_id, (image, label) in enumerate(loader()):
            ...         out = fc(image)
            ...         loss = loss_fn(out, label)
            ...         loss.backward()
            ...         adam.step()
            ...         adam.clear_grad()
            ...         print("Epoch {} batch {}: loss = {}".format(
            ...             epoch_id, batch_id, np.mean(loss.numpy())))
    """
