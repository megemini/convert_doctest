

def accuracy(input, label, k=1, correct=None, total=None):
    """

    accuracy layer.
    Refer to the https://en.wikipedia.org/wiki/Precision_and_recall
    This function computes the accuracy using the input and label.
    If the correct label occurs in top k predictions, then correct will increment by one.

    Note:
        the dtype of accuracy is determined by input. the input and label dtype can be different.

    Args:
        input(Tensor): The input of accuracy layer, which is the predictions of network. A Tensor with type float32,float64.
            The shape is ``[sample_number, class_dim]`` .
        label(Tensor): The label of dataset.  Tensor with type int32,int64. The shape is ``[sample_number, 1]`` .
        k(int, optional): The top k predictions for each class will be checked. Data type is int64 or int32. Default is 1.
        correct(Tensor, optional): The correct predictions count. A Tensor with type int64 or int32. Default is None.
        total(Tensor, optional): The total entries count. A tensor with type int64 or int32. Default is None.

    Returns:
        Tensor, The correct rate. A Tensor with type float32.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> import paddle.static as static
            >>> import paddle.nn.functional as F
            >>> paddle.seed(2023)
            >>> paddle.enable_static()
            >>> data = static.data(name="input", shape=[-1, 32, 32], dtype="float32")
            >>> label = static.data(name="label", shape=[-1,1], dtype="int")
            >>> fc_out = static.nn.fc(x=data, size=10)
            >>> predict = F.softmax(x=fc_out)
            >>> result = static.accuracy(input=predict, label=label, k=5)
            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())
            >>> np.random.seed(1107)
            >>> x = np.random.rand(3, 32, 32).astype("float32")
            >>> y = np.array([[1],[0],[1]])
            >>> output = exe.run(feed={"input": x,"label": y},
            ...                  fetch_list=[result])
            >>> print(output)

    """

def auc(
    input,
    label,
    curve='ROC',
    num_thresholds=2**12 - 1,
    topk=1,
    slide_steps=1,
    ins_tag_weight=None,
):
    """
    **Area Under the Curve (AUC) Layer**

    This implementation computes the AUC according to forward output and label.
    It is used very widely in binary classification evaluation.

    Note: If input label contains values other than 0 and 1, it will be cast
    to `bool`. Find the relevant definitions `here <https://en.wikipedia.org\
    /wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

    There are two types of possible curves:

        1. ROC: Receiver operating characteristic;
        2. PR: Precision Recall

    Args:
        input(Tensor): A floating-point 2D Tensor, values are in the range
                         [0, 1]. Each row is sorted in descending order. This
                         input should be the output of topk. Typically, this
                         Tensor indicates the probability of each label.
                         A Tensor with type float32,float64.
        label(Tensor): A 2D int Tensor indicating the label of the training
                         data. The height is batch size and width is always 1.
                         A Tensor with type int32,int64.
        curve(str, optional): Curve type, can be 'ROC' or 'PR'. Default 'ROC'.
        num_thresholds(int, optional): The number of thresholds to use when discretizing
                             the roc curve. Default 4095.
        topk(int, optional): only topk number of prediction output will be used for auc.
        slide_steps(int, optional): when calc batch auc, we can not only use step currently but the previous steps can be used. slide_steps=1 means use the current step, slide_steps=3 means use current step and the previous second steps, slide_steps=0 use all of the steps.
        ins_tag_weight(Tensor, optional): A 2D int Tensor indicating the data's tag weight, 1 means real data, 0 means fake data. Default None, and it will be assigned to a tensor of value 1.
                         A Tensor with type float32,float64.

    Returns:
        Tensor: A tuple representing the current AUC. Data type is Tensor, supporting float32, float64.
        The return tuple is auc_out, batch_auc_out, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg ]

            auc_out: the result of the accuracy rate
            batch_auc_out: the result of the batch accuracy
            batch_stat_pos: the statistic value for label=1 at the time of batch calculation
            batch_stat_neg: the statistic value for label=0 at the time of batch calculation
            stat_pos: the statistic for label=1 at the time of calculation
            stat_neg: the statistic for label=0 at the time of calculation


    Examples:
        .. code-block:: python
            :name: e0
        
            >>> import paddle
            >>> import numpy as np
            >>> paddle.enable_static()

            >>> paddle.seed(2023)
            >>> data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
            >>> label = paddle.static.data(name="label", shape=[-1], dtype="int")
            >>> fc_out = paddle.static.nn.fc(x=data, size=2)
            >>> predict = paddle.nn.functional.softmax(x=fc_out)
            >>> result=paddle.static.auc(input=predict, label=label)

            >>> place = paddle.CPUPlace()
            >>> exe = paddle.static.Executor(place)

            >>> exe.run(paddle.static.default_startup_program())
            >>> np.random.seed(1107)
            >>> x = np.random.rand(3,32,32).astype("float32")
            >>> y = np.array([1,0,1])
            >>> output= exe.run(feed={"input": x,"label": y},
            ...                 fetch_list=[result[0]])
            >>> print(output)

    Examples:
        .. code-block:: python
            :name: e1

            # you can learn the usage of ins_tag_weight by the following code.

            >>> import paddle
            >>> import numpy as np
            >>> paddle.enable_static()

            >>> paddle.seed(2023)
            >>> data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
            >>> label = paddle.static.data(name="label", shape=[-1], dtype="int")
            >>> ins_tag_weight = paddle.static.data(name='ins_tag_weight', shape=[-1,16], lod_level=0, dtype='float64')
            >>> fc_out = paddle.static.nn.fc(x=data, size=2)
            >>> predict = paddle.nn.functional.softmax(x=fc_out)
            >>> result=paddle.static.auc(input=predict, label=label, ins_tag_weight=ins_tag_weight)

            >>> place = paddle.CPUPlace()
            >>> exe = paddle.static.Executor(place)

            >>> exe.run(paddle.static.default_startup_program())
            >>> np.random.seed(1107)
            >>> x = np.random.rand(3,32,32).astype("float32")
            >>> y = np.array([1,0,1])
            >>> z = np.array([1,0,1]).astype("float64")
            >>> output= exe.run(feed={"input": x,"label": y, "ins_tag_weight":z},
            ...                 fetch_list=[result[0]])
            >>> print(output)

    """

def ctr_metric_bundle(input, label, ins_tag_weight=None):
    """
    ctr related metric layer

    This function help compute the ctr related metrics: RMSE, MAE, predicted_ctr, q_value.
    To compute the final values of these metrics, we should do following computations using
    total instance number:
    MAE = local_abserr / instance number
    RMSE = sqrt(local_sqrerr / instance number)
    predicted_ctr = local_prob / instance number
    q = local_q / instance number
    Note that if you are doing distribute job, you should all reduce these metrics and instance
    number first

    Args:
        input(Tensor): A floating-point 2D Tensor, values are in the range
                         [0, 1]. Each row is sorted in descending order. This
                         input should be the output of topk. Typically, this
                         Tensor indicates the probability of each label.
        label(Tensor): A 2D int Tensor indicating the label of the training
                         data. The height is batch size and width is always 1.
        ins_tag_weight(Tensor): A 2D int Tensor indicating the ins_tag_weight of the training
                         data. 1 means real data, 0 means fake data.
                         A LoDTensor or Tensor with type float32,float64.

    Returns:
        local_sqrerr(Tensor): Local sum of squared error
        local_abserr(Tensor): Local sum of abs error
        local_prob(Tensor): Local sum of predicted ctr
        local_q(Tensor): Local sum of q value

    Examples:
        .. code-block:: python
            :name: e0
            >>> import paddle
            >>> paddle.enable_static()
            >>> data = paddle.static.data(name="data", shape=[-1, 32], dtype="float32")
            >>> label = paddle.static.data(name="label", shape=[-1, 1], dtype="int32")
            >>> predict = paddle.nn.functional.sigmoid(paddle.static.nn.fc(x=data, size=1))
            >>> auc_out = paddle.static.ctr_metric_bundle(input=predict, label=label)

    Examples:
        .. code-block:: python
            :name: e1

            >>> import paddle
            >>> paddle.enable_static()
            >>> data = paddle.static.data(name="data", shape=[-1, 32], dtype="float32")
            >>> label = paddle.static.data(name="label", shape=[-1, 1], dtype="int32")
            >>> predict = paddle.nn.functional.sigmoid(paddle.static.nn.fc(x=data, size=1))
            >>> ins_tag_weight = paddle.static.data(name='ins_tag_weight', shape=[-1, 1], lod_level=0, dtype='int64')
            >>> auc_out = paddle.static.ctr_metric_bundle(input=predict, label=label, ins_tag_weight=ins_tag_weight)
    """



if __name__ == '__main__':
    from copy_preprocess import extract_code_blocks_from_docstr

    doc_0 = accuracy.__doc__
    doc_1 = auc.__doc__
    doc_2 = ctr_metric_bundle.__doc__

    print('-'*30)
    d = extract_code_blocks_from_docstr(doc_0)
    print(len(d))
    print('-'*30)
    d = extract_code_blocks_from_docstr(doc_1)
    print(len(d))
    print('-'*30)
    d = extract_code_blocks_from_docstr(doc_2)
    print(len(d))