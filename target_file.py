# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers

import numpy as np

import paddle
from paddle.distribution import distribution
from paddle.fluid import framework


class Cauchy(distribution.Distribution):
    r"""Cauchy distribution is also called Cauchy–Lorentz distribution. It is a continuous probability distribution named after Augustin-Louis Cauchy and Hendrik Lorentz. It has a very wide range of applications in natural sciences.

    The Cauchy distribution has the probability density function (PDF):

    .. math::

        { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

    Args:
        loc (float|Tensor): Location of the peak of the distribution. The data type is float32 or float64.
        scale (float|Tensor): The half-width at half-maximum (HWHM). The data type is float32 or float64. Must be positive values.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Cauchy

            >>> # init Cauchy with float
            >>> rv = Cauchy(loc=0.1, scale=1.2)
            >>> for i in range(3):
            ...     print(i)
            ...     print(i)
            ...     print(i)
            ...     for j in range(3):
            ...         print(j)

            >>> print(rv.entropy())
            >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        2.71334577)

            >>> # init Cauchy with N-Dim tensor
            >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
            >>> print(rv.entropy())
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [2.53102422, 3.22417140])
    """

    def __init__(self, loc, scale, name=None):
        self.name = name if name is not None else 'Cauchy'

        if not isinstance(loc, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of loc is Real|Variable, but got {type(loc)}"
            )
        if not isinstance(scale, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of scale is Real|Variable, but got {type(scale)}"
            )

        if isinstance(loc, numbers.Real):
            loc = paddle.full(shape=(), fill_value=loc)

        if isinstance(scale, numbers.Real):
            scale = paddle.full(shape=(), fill_value=scale)

        if loc.shape != scale.shape:
            self.loc, self.scale = paddle.broadcast_tensors([loc, scale])
        else:
            self.loc, self.scale = loc, scale

        self.dtype = self.loc.dtype

        super().__init__(batch_shape=self.loc.shape, event_shape=())

    @property
    def mean(self):
        """Mean of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no mean.")

    @property
    def variance(self):
        """Variance of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no variance.")

    @property
    def stddev(self):
        """Standard Deviation of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no stddev.")

    def sample(self, shape, name=None):
        """Sample from Cauchy distribution.

        Note:
            `sample` method has no grad, if you want so, please use `rsample` instead.

        Args:
            shape (Sequence[int]): Sample shape.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:

            .. code-block:: python
                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.sample([10]).shape)
                >>> # [10]

        blablablablablablablablablablablablablablablablablablablablablabla
        blablablablablablablablablablablabla
        blablablablablablablablablablablablablablablablablablablablablablablablablablablabla...

            .. code-block:: python
                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.sample([10]).shape)
                >>> # [10]

        """
        name = name if name is not None else (self.name + '_sample')
        with paddle.no_grad():
            return self.rsample(shape, name)

    def rsample(self, shape, name=None):
        """Sample from Cauchy distribution (reparameterized).

        Args:
            shape (Sequence[int]): Sample shape.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.rsample([10]).shape)
                >>> # [10]

                >>> # init Cauchy with 0-Dim tensor
                >>> rv = Cauchy(loc=paddle.full((), 0.1), scale=paddle.full((), 1.2))
                >>> print(rv.rsample([10]).shape)
                >>> # [10]

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.rsample([10]).shape)
                >>> # [10, 2]

                >>> # sample 2-Dim data
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.rsample([10, 2]).shape)
                >>> # [10, 2]

                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.rsample([10, 2]).shape)
                >>> # [10, 2, 2]
        """
        name = name if name is not None else (self.name + '_rsample')

        if not isinstance(shape, (np.ndarray, framework.Variable, list, tuple)):
            raise TypeError(
                f"Expected type of shape is Sequence[int], but got {type(shape)}"
            )

        shape = shape if isinstance(shape, tuple) else tuple(shape)
        shape = self._extend_shape(shape)

        loc = self.loc.expand(shape)
        scale = self.scale.expand(shape)
        uniforms = paddle.rand(shape, dtype=self.dtype)
        return paddle.add(
            loc,
            paddle.multiply(scale, paddle.tan(np.pi * (uniforms - 0.5))),
            name=name,
        )

    def prob(self, value):
        r"""Probability density function(PDF) evaluated at value.

        .. math::

            { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: PDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.prob(paddle.to_tensor(1.5)))
                >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        0.11234467)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.11234467, 0.01444674])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.10753712, 0.02195240])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.10753712, 0.02195240])
        """
        name = self.name + '_prob'

        if not isinstance(value, framework.Variable):
            raise TypeError(
                f"Expected type of value is Variable, but got {type(value)}"
            )

        return self.log_prob(value).exp(name=name)

    def log_prob(self, value):
        """Log of probability densitiy function.

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: Log of probability densitiy evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.log_prob(paddle.to_tensor(1.5)))
                >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        -2.18618369)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [-2.18618369, -4.23728657])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [-2.22991920, -3.81887865])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [-2.22991920, -3.81887865])
        """
        name = self.name + '_log_prob'

        if not isinstance(value, framework.Variable):
            raise TypeError(
                f"Expected type of value is Variable, but got {type(value)}"
            )

        value = self._check_values_dtype_in_probs(self.loc, value)
        loc, scale, value = paddle.broadcast_tensors(
            [self.loc, self.scale, value]
        )

        return paddle.subtract(
            -(
                paddle.square(paddle.divide(paddle.subtract(value, loc), scale))
            ).log1p(),
            paddle.add(
                paddle.full(loc.shape, np.log(np.pi), dtype=self.dtype),
                scale.log(),
            ),
            name=name,
        )

    def cdf(self, value):
        r"""Cumulative distribution function(CDF) evaluated at value.

        .. math::

            { \frac{1}{\pi} \arctan\left(\frac{x-loc}{ scale}\right)+\frac{1}{2}\! }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: CDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.cdf(paddle.to_tensor(1.5)))
                >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        0.77443725)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.77443725, 0.92502367])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.80256844, 0.87888104])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.80256844, 0.87888104])
        """
        name = self.name + '_cdf'

        if not isinstance(value, framework.Variable):
            raise TypeError(
                f"Expected type of value is Variable, but got {type(value)}"
            )

        value = self._check_values_dtype_in_probs(self.loc, value)
        loc, scale, value = paddle.broadcast_tensors(
            [self.loc, self.scale, value]
        )

        return (
            paddle.atan(
                paddle.divide(paddle.subtract(value, loc), scale), name=name
            )
            / np.pi
            + 0.5
        )

    def entropy(self):
        r"""Entropy of Cauchy distribution.

        .. math::

            { \log(4\pi scale)\! }

        Returns:
            Tensor: Entropy of distribution.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

        Examples:

            .. code-block:: python
                :name: sdfasdf-dfa-1

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.entropy())
                >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        2.71334577)

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.entropy())
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [2.53102422, 3.22417140])


            .. code-block:: python
                :name: sdfasdf-dfa-1

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.entropy())
                >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        2.71334577)

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.entropy())
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [2.53102422, 3.22417140])

        """
        name = self.name + '_entropy'
        return paddle.add(
            paddle.full(self.loc.shape, np.log(4 * np.pi), dtype=self.dtype),
            self.scale.log(),
            name=name,
        )

    def kl_divergence(self, other):
        """The KL-divergence between two Cauchy distributions.

        Note:
            [1] Frédéric Chyzak, Frank Nielsen, A closed-form formula for the Kullback-Leibler divergence between Cauchy distributions, 2019

        Args:
            other (Cauchy): instance of Cauchy.

        Returns:
            Tensor: kl-divergence between two Cauchy distributions.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> rv_other = Cauchy(loc=paddle.to_tensor(1.2), scale=paddle.to_tensor([2.3, 3.4]))
                >>> print(rv.kl_divergence(rv_other))
                >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                >>> #        [0.19819736, 0.31532931])
        """
        name = self.name + '_kl_divergence'

        if not isinstance(other, Cauchy):
            raise TypeError(
                f"Expected type of other is Cauchy, but got {type(other)}"
            )

        a_loc = self.loc
        b_loc = other.loc

        a_scale = self.scale
        b_scale = other.scale

        t1 = paddle.add(
            paddle.pow(paddle.add(a_scale, b_scale), 2),
            paddle.pow(paddle.subtract(a_loc, b_loc), 2),
        ).log()
        t2 = (4 * paddle.multiply(a_scale, b_scale)).log()

        return paddle.subtract(t1, t2, name=name)


# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
math functions
"""
# TODO: define math functions

import numpy as np

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.common_ops_import import VarDesc, dygraph_only, dygraph_utils

# TODO: define math functions
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ..common_ops_import import Variable
from ..fluid.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from ..framework import (
    LayerHelper,
    _dygraph_tracer,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
)
from .creation import _complex_to_real_dtype
from .layer_function_generator import generate_layer_fn, templatedoc
from .manipulation import cast
from .ops import abs  # noqa: F401
from .ops import acos  # noqa: F401
from .ops import acosh  # noqa: F401
from .ops import asin  # noqa: F401
from .ops import asinh  # noqa: F401
from .ops import atan  # noqa: F401
from .ops import atanh  # noqa: F401
from .ops import ceil  # noqa: F401
from .ops import ceil_  # noqa: F401
from .ops import cos  # noqa: F401
from .ops import cosh  # noqa: F401
from .ops import erf  # noqa: F401
from .ops import exp  # noqa: F401
from .ops import exp_  # noqa: F401
from .ops import expm1  # noqa: F401
from .ops import floor  # noqa: F401
from .ops import floor_  # noqa: F401
from .ops import reciprocal  # noqa: F401
from .ops import reciprocal_  # noqa: F401
from .ops import round  # noqa: F401
from .ops import round_  # noqa: F401
from .ops import rsqrt  # noqa: F401
from .ops import rsqrt_  # noqa: F401
from .ops import sigmoid  # noqa: F401
from .ops import sigmoid_  # noqa: F401
from .ops import sin  # noqa: F401
from .ops import sinh  # noqa: F401
from .ops import sqrt  # noqa: F401
from .ops import sqrt_  # noqa: F401
from .ops import square  # noqa: F401
from .ops import tan  # noqa: F401

__all__ = []

_supported_int_dtype_ = [
    VarDesc.VarType.UINT8,
    VarDesc.VarType.INT8,
    VarDesc.VarType.INT16,
    VarDesc.VarType.INT32,
    VarDesc.VarType.INT64,
]

_supported_float_dtype_ = [
    VarDesc.VarType.FP32,
    VarDesc.VarType.FP64,
]


def _get_reduce_axis(axis, x):
    """
    Internal function for max, min, amax and amin.
    It computes the attribute reduce_all value based on axis.
    """
    if axis is not None and not isinstance(axis, list):
        if isinstance(axis, (tuple, range)):
            axis = list(axis)
        elif isinstance(axis, int):
            axis = [axis]
        else:
            raise TypeError(
                "The type of axis must be int, list or tuple, but received {}".format(
                    type(axis)
                )
            )
    if axis is None:
        axis = []
    if axis == [] or len(axis) == len(x.shape):
        reduce_all = True
    else:
        reduce_all = False
    return reduce_all, axis


def _get_reduce_axis_with_tensor(axis, x):
    if isinstance(axis, Variable):
        if axis.shape[0] == len(x.shape):
            reduce_all = True
        else:
            reduce_all = False
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        if paddle.utils._contain_var(axis):
            axis = paddle.utils._convert_to_tensor_list(axis)
    return reduce_all, axis


def log(x, name=None):
    r"""
    Calculates the natural log of the given input Tensor, element-wise.

    .. math::

        Out = \ln(x)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Tensor: The natural log of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = [[2,3,4], [7,8,9]]
            >>> x = paddle.to_tensor(x, dtype='float32')
            >>> res = paddle.log(x)
            >>> # [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]
    """
    if in_dynamic_mode():
        return _C_ops.log(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'uint16', 'float16', 'float32', 'float64'],
            "log",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
        return out


def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Scale operator.

    Putting scale and bias to the input Tensor as following:

    ``bias_after_scale`` is True:

    .. math::
                            Out=scale*X+bias

    ``bias_after_scale`` is False:

    .. math::
                            Out=scale*(X+bias)

    Args:
        x (Tensor): Input N-D Tensor of scale operator. Data type can be float32, float64, int8, int16, int32, int64, uint8.
        scale (float|Tensor): The scale factor of the input, it should be a float number or a 0-D Tensor with shape [] and data type as float32.
        bias (float): The bias to be put on the input.
        bias_after_scale (bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act (str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Output Tensor of scale operator, with shape and data type same as input.

    Examples:
        .. code-block:: python

            >>> # scale as a float32 number
            >>> import paddle

            >>> data = paddle.randn(shape=[2,3], dtype='float32')
            >>> res = paddle.scale(data, scale=2.0, bias=1.0)

        .. code-block:: python

            >>> # scale with parameter scale as a Tensor
            >>> import paddle

            >>> data = paddle.randn(shape=[2, 3], dtype='float32')
            >>> factor = paddle.to_tensor([2], dtype='float32')
            >>> res = paddle.scale(data, scale=factor, bias=1.0)

    """

    if in_dynamic_mode():
        if act is None:
            return _C_ops.scale(x, scale, float(bias), bias_after_scale)
        out = _C_ops.scale(x, scale, float(bias), bias_after_scale)
        return dygraph_utils._append_activation_in_dygraph(out, act)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'float16',
                'uint16',
                'float32',
                'float64',
                'int8',
                'int16',
                'int32',
                'int64',
                'uint8',
            ],
            "scale",
        )
        inputs = {'X': [x]}
        attrs = {
            'bias': float(bias),
            'bias_after_scale': bias_after_scale,
        }
        if isinstance(scale, Variable):
            inputs['ScaleTensor'] = [scale]
        else:
            attrs['scale'] = float(scale)
        helper = LayerHelper('scale', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='scale', inputs=inputs, outputs={'Out': out}, attrs=attrs
        )
        return helper.append_activation(out)


def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    r"""

    stanh activation.

    .. math::

        out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        scale_a (float, optional): The scale factor a of the input. Default is 0.67.
        scale_b (float, optional): The scale factor b of the output. Default is 1.7159.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
            >>> out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]

    """

    if in_dynamic_mode():
        return _C_ops.stanh(x, scale_a, scale_b)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'stanh'
        )

        helper = LayerHelper('stanh', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='stanh',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'scale_a': scale_a, 'scale_b': scale_b},
        )
        return out


def multiplex(inputs, index, name=None):
    """

    Based on the given index parameter, the OP selects a specific row from each input Tensor to construct the output Tensor.

    If the input of this OP contains :math:`m` Tensors, where :math:`I_{i}` means the i-th input Tensor, :math:`i` between :math:`[0,m)` .

    And :math:`O` means the output, where :math:`O[i]` means the i-th row of the output, then the output satisfies that :math:`O[i] = I_{index[i]}[i]` .

    For Example:

            .. code-block:: text

                Given:

                inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                          [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                          [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                          [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

                index = [[3],[0],[1],[2]]

                out = [[3,0,3,4],    # out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
                       [0,1,3,4],    # out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
                       [1,2,4,2],    # out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
                       [2,3,3,4]]    # out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]


    Args:
        inputs (list): The input Tensor list. The list elements are N-D Tensors of data types float32, float64, int32, int64. All input Tensor shapes should be the same and rank must be at least 2.
        index (Tensor): Used to select some rows in the input Tensor to construct an index of the output Tensor. It is a 2-D Tensor with data type int32 or int64 and shape [M, 1], where M is the number of input Tensors.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Output of multiplex OP, with data type being float32, float64, int32, int64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> img1 = paddle.to_tensor([[1, 2], [3, 4]], dtype=paddle.float32)
            >>> img2 = paddle.to_tensor([[5, 6], [7, 8]], dtype=paddle.float32)
            >>> inputs = [img1, img2]
            >>> index = paddle.to_tensor([[1], [0]], dtype=paddle.int32)
            >>> res = paddle.multiplex(inputs, index)
            >>> print(res) # Tensor([[5., 6.], [3., 4.]], dtype=float32)

    """
    if in_dynamic_mode():
        return _C_ops.multiplex(inputs, index)
    else:
        helper = LayerHelper('multiplex', **locals())

        check_type(inputs, 'inputs', (list), 'multiplex')
        if len(inputs) < 2:
            raise ValueError(
                "inputs should be a list object with at least 2 elements."
            )
        for id, x in enumerate(inputs):
            check_variable_and_dtype(
                x,
                'input[' + str(id) + ']',
                ['float32', 'float64', 'int32', 'int64'],
                'multiplex',
            )
        check_variable_and_dtype(
            index, "index", ['int32', 'int64'], 'multiplex'
        )

        out = helper.create_variable_for_type_inference(inputs[0].dtype)
        helper.append_op(
            type='multiplex',
            inputs={'X': inputs, 'Ids': index},
            outputs={'Out': [out]},
        )
        return out


@inplace_apis_in_dygraph_only
def scale_(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Inplace version of ``scale`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_scale`.
    """
    if in_dynamic_mode():
        return _C_ops.scale_(x, scale, float(bias), bias_after_scale)


def pow(x, y, name=None):
    """
    Compute the power of Tensor elements. The equation is:

    .. math::
        out = x^{y}

    Note:
        ``paddle.pow`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensors


    Args:
        x (Tensor): An N-D Tensor, the data type is float16, float32, float64, int32 or int64.
        y (float|int|Tensor): If it is an N-D Tensor, its data type should be the same as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. Its dimension and data type are the same as `x`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> # example 1: y is a float or int
            >>> res = paddle.pow(x, 2)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [1., 4., 9.])
            >>> res = paddle.pow(x, 2.5)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [1.         , 5.65685415 , 15.58845711])

            >>> # example 2: y is a Tensor
            >>> y = paddle.to_tensor([2], dtype='float32')
            >>> res = paddle.pow(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [1., 4., 9.])

    """
    # in dynamic graph mode
    if in_dynamic_mode():
        if isinstance(y, (int, float)):
            return _C_ops.pow(x, y)
        elif isinstance(y, (paddle.Tensor, Variable)):
            return _C_ops.elementwise_pow(x, y)
        else:
            raise TypeError(
                'y must be scalar or tensor type, but received: %s ' % (y.dtype)
            )
    else:
        # in static graph mode
        if isinstance(y, (int, float)):
            helper = LayerHelper('pow', **locals())
            inputs = {'X': x}
            attrs = {'factor': y}
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs
            )
            return out
        elif isinstance(y, (paddle.Tensor, Variable)):
            # TODO A potential speed improvement is supporting different types in C++ and removing the cast ops here
            helper = LayerHelper('elementwise_pow', **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            return _elementwise_op(LayerHelper('elementwise_pow', **locals()))
        else:
            raise TypeError(
                'y must be scalar or tensor type, but received: %s ' % (type(y))
            )


@inplace_apis_in_dygraph_only
def pow_(x, y, name=None):
    """
    Inplace version of ``pow`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_paddle_pow`.
    """
    if isinstance(y, (int, float)):
        return _C_ops.pow_(x, y)
    elif isinstance(y, (paddle.Tensor, Variable)):
        return _C_ops.elementwise_pow_(x, y)
    else:
        raise TypeError(
            'y must be scalar or tensor type, but received: %s ' % (type(y))
        )


OP_NAMEMAPPING = {
    'elementwise_max': 'maximum',
    'elementwise_min': 'minimum',
    'elementwise_pow': 'elementwise_pow',
    'elementwise_floordiv': 'floor_divide',
    'elementwise_add': 'add',
    'elementwise_sub': 'subtract',
    'elementwise_mul': 'multiply',
    'elementwise_div': 'divide',
    'elementwise_mod': 'remainder',
}


def _elementwise_op(helper):
    op_type = helper.layer_type
    original_op_type = helper.kwargs.get('original_op_type', op_type)
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    out = helper.kwargs.get('out', None)

    assert x is not None, f'x cannot be None in {original_op_type}'
    assert y is not None, f'y cannot be None in {original_op_type}'
    bf16_and_complex_supported_ops = [
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "elementwise_div",
        "elementwise_max",
    ]
    if original_op_type in bf16_and_complex_supported_ops:
        data_type = [
            'uint16',
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
            'complex64',
            'complex128',
        ]
    else:
        data_type = [
            'float16',
            'uint16',
            'float32',
            'float64',
            'int32',
            'int64',
            'bool',
        ]
    check_variable_and_dtype(
        x,
        'x',
        data_type,
        original_op_type,
    )
    check_variable_and_dtype(
        y,
        'y',
        data_type,
        original_op_type,
    )

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False
            )

    helper.append_op(
        type=op_type,
        inputs={'X': x, 'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis, 'use_mkldnn': use_mkldnn},
    )
    return helper.append_activation(out)


def add(x, y, name=None):
    """
    Elementwise Add Operator.
    Add two tensors element-wise
    The equation is:

    ..  math::

        Out=X+Y

    $X$ the tensor of any dimension.
    $Y$ the tensor whose dimensions must be less than or equal to the dimensions of $X$.

    There are two cases for this operator:

    1. The shape of $Y$ is the same with $X$.
    2. The shape of $Y$ is a continuous subsequence of $X$.

    For case 2:

    1. Broadcast $Y$ to match the shape of $X$, where axis is the start dimension index for broadcasting $Y$ onto $X$.
    2. If $axis$ is -1 (default), $axis$=rank($X$)-rank($Y$).
    3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of subsequence, such as shape($Y$) = (2, 1) => (2).

        For example:

        .. code-block:: python

            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

    Args:
        x (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
        y (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
        name (string, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with x.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 4], 'float64')
            >>> y = paddle.to_tensor([1, 5, 2], 'float64')
            >>> z = paddle.add(x, y)
            >>> print(z)  # [3., 8., 6. ]
    """

    if in_dynamic_mode():
        return _C_ops.add(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_add', **locals()))


@inplace_apis_in_dygraph_only
def add_(x, y, name=None):
    """
    Inplace version of ``add`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_add`.
    """

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.add_(x, y)


def logaddexp(x, y, name=None):
    """
    Elementwise LogAddExp Operator.
    Add of exponentiations of the inputs
    The equation is:

    ..  math::

        Out=log(X.exp()+Y.exp())

    $X$ the tensor of any dimension.
    $Y$ the tensor whose dimensions must be less than or equal to the dimensions of $X$.

    There are two cases for this operator:

    1. The shape of $Y$ is the same with $X$.
    2. The shape of $Y$ is a continuous subsequence of $X$.

    For case 2:

    1. Broadcast $Y$ to match the shape of $X$, where axis is the start dimension index for broadcasting $Y$ onto $X$.
    2. If $axis$ is -1 (default), $axis$=rank($X$)-rank($Y$).
    3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of subsequence, such as shape($Y$) = (2, 1) => (2).

        For example:

        .. code-block:: python

            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
            >>> shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

    Args:
        x (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64, float16.
        y (Tensor): Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64, float16.
        name (string, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with x.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1, -2, -3], 'float64')
            >>> y = paddle.to_tensor([-1], 'float64')
            >>> z = paddle.logaddexp(x, y)
            >>> print(z)  # [-0.30685282, -0.68673831, -0.87307199]
    """

    return paddle.log1p(paddle.exp(-paddle.abs(x - y))) + paddle.maximum(x, y)


def subtract(x, y, name=None):
    """
    Substract two tensors element-wise. The equation is:

    .. math::
        out = x - y

    Note:
        ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[5, 6], [3, 4]])
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[-4, -4],
            >>> #         [ 4,  4]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([1, 0, 4])
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            >>> # Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[[ 0,  2, -1],
            >>> #          [ 0,  2, -1]]])

            >>> x = paddle.to_tensor([2, float('nan'), 5], dtype='float32')
            >>> y = paddle.to_tensor([1, 4, float('nan')], dtype='float32')
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [1. , nan, nan])

            >>> x = paddle.to_tensor([5, float('inf'), -float('inf')], dtype='float64')
            >>> y = paddle.to_tensor([1, 4, 5], dtype='float64')
            >>> res = paddle.subtract(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        [ 4.  ,  inf., -inf.])
    """
    if in_dynamic_mode():
        return _C_ops.subtract(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


@inplace_apis_in_dygraph_only
def subtract_(x, y, name=None):
    """
    Inplace version of ``subtract`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_subtract`.
    """

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.subtract_(x, y)


def divide(x, y, name=None):
    """
    Divide two tensors element-wise. The equation is:

    .. math::
        out = x / y

    Note:
        ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 4], dtype='float64')
            >>> y = paddle.to_tensor([1, 5, 2], dtype='float64')
            >>> z = paddle.divide(x, y)
            >>> print(z)  # [2., 0.6, 2.]

    """
    if in_dynamic_mode():
        return _C_ops.divide(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_div', **locals()))


def floor_divide(x, y, name=None):
    """
    Floor divide two tensors element-wise and rounds the quotinents to the nearest integer toward zero. The equation is:

    .. math::
        out = trunc(x / y)

    - :math:`x`: Multidimensional Tensor.
    - :math:`y`: Multidimensional Tensor.

    Note:
        ``paddle.floor_divide`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

        Also note that the name ``floor_divide`` can be misleading, as the quotinents are actually rounded toward zero, not toward negative infinite.

    Args:
        x (Tensor): the input tensor, it's data type should be uint8, int8, int32, int64, float32, float64, float16, bfloat16.
        y (Tensor): the input tensor, it's data type should be uint8, int8, int32, int64, float32, float64, float16, bfloat16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. It's dimension equals with $x$.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 8, 7])
            >>> y = paddle.to_tensor([1, 5, 3, 3])
            >>> z = paddle.floor_divide(x, y)
            >>> print(z)  # [2, 0, 2, 2]

    """
    if in_dynamic_mode():
        return _C_ops.floor_divide(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_floordiv', **locals()))


def remainder(x, y, name=None):
    r"""
    Mod two tensors element-wise. The equation is:

    .. math::

        out = x \% y

    Note:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([2, 3, 8, 7])
            >>> y = paddle.to_tensor([1, 5, 3, 3])
            >>> z = paddle.remainder(x, y)
            >>> print(z)  # [0, 3, 2, 1]

    """
    if in_dynamic_mode():
        return _C_ops.remainder(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_mod', **locals()))


@inplace_apis_in_dygraph_only
def remainder_(x, y, name=None):
    r"""
    Inplace version of ``remainder`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_remainder`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.remainder_(x, y)


mod = remainder  # noqa: F841
floor_mod = remainder  # noqa: F841


def multiply(x, y, name=None):
    """
    multiply two tensors element-wise. The equation is:

    .. math::
        out = x * y

    Note:
        ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [3, 4]])
            >>> y = paddle.to_tensor([[5, 6], [7, 8]])
            >>> res = paddle.multiply(x, y)
            >>> print(res) # [[5, 12], [21, 32]]

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([2])
            >>> res = paddle.multiply(x, y)
            >>> print(res) # [[[2, 4, 6], [2, 4, 6]]]

    """
    if in_dynamic_mode():
        return _C_ops.multiply(x, y)
    else:
        if x.dtype != y.dtype:
            raise TypeError(
                f'Input tensors must be same type, but received type of x: {x.dtype}, type of y: {y.dtype} '
            )

        return _elementwise_op(LayerHelper('elementwise_mul', **locals()))


@inplace_apis_in_dygraph_only
def multiply_(x, y, name=None):
    """
    Inplace version of ``multiply`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_multiply`.
    """

    assert (
        _dygraph_tracer()._has_grad is False
    ), "The current inplace version of multiply_ needs to be used in the context of paddle.no_grad() since inplace multiply_grad is not yet supported."

    out_shape = broadcast_shape(x.shape, y.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )

    return _C_ops.multiply_(x, y)


@dygraph_only
def _elementwise_op_with_axis_in_dygraph(
    x, y, axis=-1, name=None, op_type="Undifined"
):
    assert (
        in_dynamic_mode()
    ), "You can only call `_elementwise_op_with_axis_in_dygraph` function within in_dynamic_mode"
    assert op_type in ["add", "subtract", "multiply", "divide"], (
        "op_name input error! _elementwise_op_with_axis is an inner function to replace elementwise_add/sub/mul/div. Input op_name=%s, Expect op_name=[add|subtract|multiply|divide]\n"
        % op_type
    )
    op = getattr(_C_ops, op_type)
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    if axis == -1 or len(x_shape) == len(y_shape):
        return op(x, y)
    if len(x_shape) > len(y_shape):
        padding = len(x_shape) - len(y_shape) - axis
        y = paddle.reshape(y, [1] * axis + y_shape + [1] * padding)
    else:
        padding = len(y_shape) - len(x_shape) - axis
        x = paddle.reshape(x, [1] * axis + y_shape + [1] * padding)
    return op(x, y)


def _add_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(x, y, axis, name, "add")
    else:
        op_type = 'elementwise_add'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _subtract_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(
            x, y, axis, name, "subtract"
        )
    else:
        op_type = 'elementwise_sub'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _multiply_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(
            x, y, axis, name, "multiply"
        )
    else:
        op_type = 'elementwise_mul'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def _divide_with_axis(x, y, axis=-1, name=None):
    # opt performance, only dynamic mode needs reshape
    if in_dynamic_mode():
        return _elementwise_op_with_axis_in_dygraph(x, y, axis, name, "divide")
    else:
        op_type = 'elementwise_div'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def maximum(x, y, name=None):
    """
    Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:

    .. math::
        out = max(x, y)

    Note:
        ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to  `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[3, 4],
            >>> #         [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[3, 2, 4],
            >>> #         [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [2. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.maximum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [5.  , 3.  , inf.])
    """
    if in_dynamic_mode():
        return _C_ops.maximum(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_max', **locals()))


def minimum(x, y, name=None):
    """
    Compare two tensors and return a new tensor containing the element-wise minima. The equation is:

    .. math::
        out = min(x, y)

    Note:
        ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[1, 2],
            >>> #         [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[[1, 0, 3],
            >>> #          [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [1. , nan, nan])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.minimum(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        [ 1.  , -inf.,  5.  ])
    """
    if in_dynamic_mode():
        return _C_ops.minimum(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_min', **locals()))


def fmax(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the maximum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmax(x, y)

    Note:
        ``paddle.fmax`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[3, 4],
            >>> #         [7, 8]])

            >>> x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[3, 2, 4],
            >>> #         [3, 2, 4]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [2., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
            >>> res = paddle.fmax(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [5.  , 3.  , inf.])
    """
    if in_dynamic_mode():
        return _C_ops.fmax(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_fmax', **locals()))


def fmin(x, y, name=None):
    """
    Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the minimum value of the element.
    If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
    The equation is:

    .. math::
        out = fmin(x, y)

    Note:
        ``paddle.fmin`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        y (Tensor): the input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2], [7, 8]])
            >>> y = paddle.to_tensor([[3, 4], [5, 6]])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[1, 2],
            >>> #         [5, 6]])

            >>> x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
            >>> y = paddle.to_tensor([3, 0, 4])
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            >>> # Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[[1, 0, 3],
            >>> #          [1, 0, 3]]])

            >>> x = paddle.to_tensor([2, 3, 5], dtype='float32')
            >>> y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [1., 3., 5.])

            >>> x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
            >>> y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
            >>> res = paddle.fmin(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        [ 1.  , -inf.,  5.  ])
    """
    if in_dynamic_mode():
        return _C_ops.fmin(x, y)
    else:
        return _elementwise_op(LayerHelper('elementwise_fmin', **locals()))


def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        if `x.dtype='bool'`, `x.dtype='int32'`, it's data type is `'int64'`,
        otherwise it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[0.2, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, 0.6, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                         [0.1, 0.2, 0.6, 0.7]])
            >>> out1 = paddle.sum(x)          # 3.5
            >>> out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
            >>> out3 = paddle.sum(x, axis=-1) # [1.9, 1.6]
            >>> out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

            >>> # y is a Tensor with shape [2, 2, 2] and elements as below:
            >>> #      [[[1, 2], [3, 4]],
            >>> #      [[5, 6], [7, 8]]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> y = paddle.to_tensor([[[1, 2], [3, 4]],
            ...                         [[5, 6], [7, 8]]])
            >>> out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
            >>> out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]

            >>> # x is a Tensor with following elements:
            >>> #    [[True, True, True, True]
            >>> #     [False, False, False, False]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[True, True, True, True],
            ...                         [False, False, False, False]])
            >>> out7 = paddle.sum(x)          # 4
            >>> out8 = paddle.sum(x, axis=0)  # [1, 1, 1, 1]
            >>> out9 = paddle.sum(x, axis=1)  # [4, 0]
    """

    dtype_flag = False
    if dtype is not None:
        dtype_flag = True
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.sum(x, axis, dtype, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}

        if dtype_flag:
            attrs.update({'in_dtype': x.dtype, 'out_dtype': dtype})

        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'uint16',
                'float16',
                'float32',
                'float64',
                'int16',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'sum',
        )

        check_type(
            axis, 'axis', (int, list, tuple, type(None), Variable), 'sum'
        )

        helper = LayerHelper('sum', **locals())
        if dtype_flag:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        else:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_sum',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None):
    """
    Replaces NaN, positive infinity, and negative infinity values in input tensor.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        nan (float, optional): the value to replace NaNs with. Default is 0.
        posinf (float, optional): if a Number, the value to replace positive infinity values with. If None, positive infinity values are replaced with the greatest finite value representable by input’s dtype. Default is None.
        neginf (float, optional): if a Number, the value to replace negative infinity values with. If None, negative infinity values are replaced with the lowest finite value representable by input’s dtype. Default is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of nan_to_num operation input Tensor ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('nan'), 0.3, float('+inf'), float('-inf')], dtype='float32')
            >>> out1 = paddle.nan_to_num(x)  # [0, 0.3, 3.4028235e+38, -3.4028235e+38]
            >>> out2 = paddle.nan_to_num(x, nan=1)  # [1, 0.3, 3.4028235e+38, -3.4028235e+38]
            >>> out3 = paddle.nan_to_num(x, posinf=5)  # [0, 0.3, 5, -3.4028235e+38]
            >>> out4 = paddle.nan_to_num(x, nan=10, neginf=-99)  # [10, 0.3, 3.4028235e+38, -99]
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    posinf_value = paddle.full_like(x, float("+inf"))
    neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    assert x.dtype in [paddle.float32, paddle.float64]
    is_float32 = x.dtype == paddle.float32
    if posinf is None:
        posinf = (
            np.finfo(np.float32).max if is_float32 else np.finfo(np.float64).max
        )
    posinf = paddle.full_like(x, posinf)
    if neginf is None:
        neginf = (
            np.finfo(np.float32).min if is_float32 else np.finfo(np.float64).min
        )
    neginf = paddle.full_like(x, neginf)
    x = paddle.where(paddle.isnan(x), nan, x)
    x = paddle.where(x == posinf_value, posinf, x)
    x = paddle.where(x == neginf_value, neginf, x)
    return x


def nansum(x, axis=None, dtype=None, keepdim=False, name=None):
    """
    Computes the sum of tensor elements over the given axis, treating Not a Numbers (NaNs) as zero.

    Args:
        x (Tensor): An N-D Tensor, the data type is float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the nansum is performed. If
            :attr:`None`, nansum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of summation operation on the specified axis of input Tensor `x`,

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a Tensor with following elements:
            >>> #    [[nan, 0.3, 0.5, 0.9]
            >>> #     [0.1, 0.2, -nan, 0.7]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
            ...                 [0.1, 0.2, float('-nan'), 0.7]],dtype="float32")
            >>> out1 = paddle.nansum(x)          # 2.7
            >>> out2 = paddle.nansum(x, axis=0)  # [0.1, 0.5, 0.5, 1.6]
            >>> out3 = paddle.nansum(x, axis=-1) # [1.7, 1.0]
            >>> out4 = paddle.nansum(x, axis=1, keepdim=True)  # [[1.7], [1.0]]

            >>> # y is a Tensor with shape [2, 2, 2] and elements as below:
            >>> #      [[[1, nan], [3, 4]],
            >>> #      [[5, 6], [-nan, 8]]]
            >>> # Each example is followed by the corresponding output tensor.
            >>> y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
            ...                 [[5, 6], [float('-nan'), 8]]])
            >>> out5 = paddle.nansum(y, axis=[1, 2]) # [8, 19]
            >>> out6 = paddle.nansum(y, axis=[0, 1]) # [9, 18]
    """
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'nansum'
    )
    check_type(axis, 'axis', (int, list, tuple, type(None)), 'nansum')

    zero_tensor = paddle.zeros_like(x)
    tmp_tensor = paddle.where(isnan(x), zero_tensor, x)
    return sum(tmp_tensor, axis, dtype, keepdim, name)


def nanmean(x, axis=None, keepdim=False, name=None):
    r"""
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Args:
        x (Tensor): The input Tensor with data type uint16, float16, float32, float64.
        axis (int|list|tuple, optional):The axis along which to perform nanmean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), nanmean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, nanmean is
            calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of arithmetic mean along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:

        .. code-block:: python
            :name: code-example1

            >>> import paddle
            >>> # x is a 2-D Tensor:
            >>> x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
            ...                         [0.1, 0.2, float('-nan'), 0.7]])
            >>> out1 = paddle.nanmean(x)
            >>> # 0.44999996
            >>> out2 = paddle.nanmean(x, axis=0)
            >>> # [0.1, 0.25, 0.5, 0.79999995]
            >>> out3 = paddle.nanmean(x, axis=0, keepdim=True)
            >>> # [[0.1, 0.25, 0.5, 0.79999995]]
            >>> out4 = paddle.nanmean(x, axis=1)
            >>> # [0.56666666 0.33333334]
            >>> out5 = paddle.nanmean(x, axis=1, keepdim=True)
            >>> # [[0.56666666]
            >>> #  [0.33333334]]

            >>> # y is a 3-D Tensor:
            >>> y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
            ...                         [[5, 6], [float('-nan'), 8]]])
            >>> out6 = paddle.nanmean(y, axis=[1, 2])
            >>> # [2.66666675, 6.33333349]
            >>> out7 = paddle.nanmean(y, axis=[0, 1])
            >>> # [3., 6.]
    """
    if isinstance(axis, int):
        axis = [axis]
    check_variable_and_dtype(
        x, 'x/input', ['uint16', 'float16', 'float32', 'float64'], 'nanmean'
    )
    if axis is not None:
        check_type(axis, 'axis/dim', (int, list, tuple), 'nanmean')

    cnt = paddle.sum(~paddle.isnan(x), axis=axis, keepdim=keepdim)
    return paddle.divide(
        paddle.nansum(x, axis=axis, keepdim=keepdim, name=name),
        cnt.astype(x.dtype),
    )


def count_nonzero(x, axis=None, keepdim=False, name=None):
    r"""
    Counts the number of non-zero values in the tensor x along the specified axis.

    Args:
        x (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
        axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results of count operation on the specified axis of input Tensor `x`, it's data type is `'int64'`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> # x is a 2-D Tensor:
            >>> x = paddle.to_tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
            >>> out1 = paddle.count_nonzero(x)
            >>> # 3
            >>> out2 = paddle.count_nonzero(x, axis=0)
            >>> # [0, 1, 2]
            >>> out3 = paddle.count_nonzero(x, axis=0, keepdim=True)
            >>> # [[0, 1, 2]]
            >>> out4 = paddle.count_nonzero(x, axis=1)
            >>> # [2, 1, 0]
            >>> out5 = paddle.count_nonzero(x, axis=1, keepdim=True)
            >>> #[[2],
            >>> # [1],
            >>> # [0]]

            >>> # y is a 3-D Tensor:
            >>> y = paddle.to_tensor([[[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]],
            ...                         [[0., 2.5, 2.6], [0., 0., 2.4], [2.1, 2.2, 2.3]]])
            >>> out6 = paddle.count_nonzero(y, axis=[1, 2])
            >>> # [3, 6]
            >>> out7 = paddle.count_nonzero(y, axis=[0, 1])
            >>> # [1, 3, 5]
    """

    if isinstance(axis, int):
        axis = [axis]

    bool_tensor = paddle.cast(x, 'bool')
    int_tensor = paddle.cast(bool_tensor, 'int64')
    return paddle.sum(int_tensor, axis=axis, keepdim=keepdim, name=name)


@templatedoc(op_type="sum")
def add_n(inputs, name=None):
    """
    Sum one or more Tensor of the input.

    For example:

    .. code-block:: text

        Case 1:

            Input:
                input.shape = [2, 3]
                input = [[1, 2, 3],
                         [4, 5, 6]]

            Output:
                output.shape = [2, 3]
                output = [[1, 2, 3],
                          [4, 5, 6]]

        Case 2:

            Input:
                First input:
                    input1.shape = [2, 3]
                    Input1 = [[1, 2, 3],
                              [4, 5, 6]]

                The second input:
                    input2.shape = [2, 3]
                    input2 = [[7, 8, 9],
                              [10, 11, 12]]

                Output:
                    output.shape = [2, 3]
                    output = [[8, 10, 12],
                              [14, 16, 18]]

    Args:
        inputs (Tensor|list[Tensor]|tuple[Tensor]):  A Tensor or a list/tuple of Tensors. The shape and data type of the list/tuple elements should be consistent.
            Input can be multi-dimensional Tensor, and data types can be: float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the sum of input :math:`inputs` , its shape and data types are consistent with :math:`inputs`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
            >>> input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
            >>> output = paddle.add_n([input0, input1])
            >>> # [[8., 10., 12.],
            >>> #  [14., 16., 18.]]
    """
    if in_dynamic_mode():
        if isinstance(inputs, Variable):
            inputs = [inputs]
        return _C_ops.add_n(inputs)
    else:
        helper = LayerHelper('add_n', **locals())
        check_type(inputs, 'inputs', (Variable, tuple, list), 'add_n')
        if isinstance(inputs, (list, tuple)):
            if len(inputs) > 0:
                for input in inputs:
                    check_variable_and_dtype(
                        input,
                        "inputs",
                        [
                            'float16',
                            'float32',
                            'float64',
                            'int32',
                            'int64',
                            'uint16',
                        ],
                        'add_n',
                    )
        else:
            check_variable_and_dtype(
                inputs,
                "inputs",
                ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
                'add_n',
            )

        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype('inputs')
        )
        helper.append_op(
            type='sum',
            inputs={'X': inputs},
            outputs={'Out': out},
            attrs={'use_mkldnn': False},
        )

        return out


def trunc(input, name=None):
    '''
    This API is used to returns a new tensor with the truncated integer values of input.

    Args:
        input (Tensor): The input tensor, it's data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor of trunc.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand([2,2],'float32')
            >>> print(input)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #         [[0.02331470, 0.42374918],
            >>> #         [0.79647720, 0.74970269]])

            >>> output = paddle.trunc(input)
            >>> print(output)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #         [[0., 0.],
            >>> #         [0., 0.]]))
    '''
    if in_dynamic_mode():
        return _C_ops.trunc(input)
    else:
        inputs = {"X": input}
        attrs = {}

        helper = LayerHelper("trunc", **locals())
        check_variable_and_dtype(
            input, 'X', ['int32', 'int64', 'float32', 'float64'], 'trunc'
        )
        out = helper.create_variable_for_type_inference(dtype=input.dtype)

        helper.append_op(
            type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


def mm(input, mat2, name=None):
    """

    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.


    Also note that if the raw tensor :math:`x` or :math:`mat2` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        input (Tensor): The input tensor which is a Tensor.
        mat2 (Tensor): The input tensor which is a Tensor.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The product Tensor.

    ::

        * example 1:

        input: [B, ..., M, K], mat2: [B, ..., K, N]
        out: [B, ..., M, N]

        * example 2:

        input: [B, M, K], mat2: [B, K, N]
        out: [B, M, N]

        * example 3:

        input: [B, M, K], mat2: [K, N]
        out: [B, M, N]

        * example 4:

        input: [M, K], mat2: [K, N]
        out: [M, N]

        * example 5:

        input: [B, M, K], mat2: [K]
        out: [B, M]

        * example 6:

        input: [K], mat2: [K]
        out: [1]

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
            >>> mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
            >>> out = paddle.mm(input, mat2)
            >>> print(out)
            >>> #        [[11., 14., 17., 20.],
            >>> #         [23., 30., 37., 44.],
            >>> #         [35., 46., 57., 68.]])


    """
    if in_dynamic_mode():
        return _C_ops.matmul(input, mat2, False, False)
    else:

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val, name, ['float16', 'float32', 'float64'], 'mm'
                )
            x_shape = list(x.shape)
            y_shape = list(y.shape)
            if len(x_shape) == 1:
                x_shape = [1] + x_shape
            if len(y_shape) == 1:
                y_shape = y_shape + [1]

            # check the inner 2 dimensions
            if x_shape[-1] != y_shape[-2]:
                if not ((x_shape[-1] == -1) or (y_shape[-2] == -1)):
                    raise ValueError(
                        "After performing an optional transpose, Input X's width should be "
                        "equal to Y's width for multiplication "
                        "prerequisites. But received X's shape: {}, Y's shape: {}\n".format(
                            x_shape, y_shape
                        )
                    )

            if len(y_shape) > 2 and len(x_shape) > 2:
                for i, dim_x in enumerate(x_shape[:-2]):
                    # don't check neg shape
                    if dim_x < 0 or y_shape[i] < 0:
                        continue
                    if dim_x != y_shape[i]:
                        raise ValueError(
                            "When the matrix is larger than 2 dimensions, the higher "
                            "dimensional values of the two matrices need to be equal. "
                            "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                            "Y's shape: %s.\n" % (i, i, x_shape, y_shape)
                        )

        __check_input(input, mat2)

        helper = LayerHelper('mm', **locals())
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type='matmul_v2',
            inputs={'X': input, 'Y': mat2},
            outputs={'Out': out},
        )
        return out


def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    """
    **addmm**

    Perform matrix multiplication for input $x$ and $y$.
    $input$ is added to the final result.
    The equation is:

    ..  math::
        Out = alpha * x * y + beta * input

    $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.

    Args:
        input (Tensor): The input Tensor to be added to the final result.
        x (Tensor): The first input Tensor for matrix multiplication.
        y (Tensor): The second input Tensor for matrix multiplication.
        beta (float, optional): Coefficient of $input$, default is 1.
        alpha (float, optional): Coefficient of $x*y$, default is 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor of addmm.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.ones([2,2])
            >>> y = paddle.ones([2,2])
            >>> input = paddle.ones([2,2])

            >>> out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

            >>> print(out)
            >>> # [[10.5 10.5]
            >>> # [10.5 10.5]]
    """
    input_shape = input.shape
    x_shape = x.shape
    y_shape = y.shape
    if not len(x_shape) == len(y_shape) == 2:
        raise ValueError(
            "The dimention of x, y should be 2 but receive x's shape: {}, y's shape: {}".format(
                x_shape, y_shape
            )
        )
    if x_shape[1] != y_shape[0]:
        raise ValueError(
            "The input Variable x's width must be equal with Variable y' height. But received x's shape = {}, y's shape = {}.".format(
                x_shape, y_shape
            )
        )
    if len(input_shape) == 2:
        if input_shape[0] != x_shape[0]:
            if input_shape[0] != 1:
                raise ValueError(
                    "When x's dimension[0] is not equal with input's dimension[0], input's dimension[0] must be 1 but got {}".format(
                        input_shape[0]
                    )
                )
            if input_shape[1] != y_shape[1] and input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
        if input_shape[1] != y_shape[1]:
            if input_shape[1] != 1:
                raise ValueError(
                    "When y's dimension[1] is not equal with input's dimension[1], input's dimension[1] must be 1 but got {}".format(
                        input_shape[1]
                    )
                )
    elif len(input_shape) == 1:
        if input_shape[0] not in (y_shape[1], 1):
            raise ValueError(
                "The input's shape: {} is not broadcastable with [x.shape[0], y.shape[1]]: [{},{}]".format(
                    input_shape, x_shape[0], y_shape[1]
                )
            )
    else:
        raise ValueError(
            "The dimention of input should be 2 or 1 but receive input's shape: {}".format(
                input_shape
            )
        )

    if in_dynamic_mode():
        return _C_ops.addmm(input, x, y, beta, alpha)
    else:
        inputs = {'Input': input, "X": x, "Y": y}
        attrs = {'Alpha': alpha, 'Beta': beta}

        helper = LayerHelper("addmm", **locals())
        check_variable_and_dtype(
            input, 'Input', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        check_variable_and_dtype(
            x, 'X', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        check_variable_and_dtype(
            y, 'Y', ['float16', 'float32', 'float64', 'uint16'], 'addmm'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="addmm", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


def renorm(x, p, axis, max_norm):
    """
    **renorm**

    This operator is used to calculate the p-norm along the axis,
    suppose the input-shape on axis dimension has the value of T, then
    the tensor is split into T parts, the p-norm should be calculated for each
    part, if the p-norm for part i is larger than max-norm, then each element
    in part i should be re-normalized at the same scale so that part-i' p-norm equals
    max-norm exactly, otherwise part-i stays unchanged.

    Args:
        x (Tensor): The input Tensor
        p (float): The power of the norm operation.
        axis (int): the dimension to slice the tensor.
        max-norm (float): the maximal norm limit.

    Returns:
        Tensor: the renorm Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> input = [[[2.0,2,-2],[3,0.3,3]],[[2,-8,2],[3.1,3.7,3]]]
            >>> x = paddle.to_tensor(input,dtype='float32')
            >>> y = paddle.renorm(x, 1.0, 2, 2.05)
            >>> print(y)
    #        [[[ 0.40594056,  0.29285714, -0.41000000],
    #          [ 0.60891086,  0.04392857,  0.61500001]],
    #         [[ 0.40594056, -1.17142856,  0.41000000],
    #          [ 0.62920785,  0.54178572,  0.61500001]]])

    """
    input_shape = x.shape
    if not axis < len(input_shape):
        raise ValueError(
            "the axis:{} should be less then the shape's size {}:{}".format(
                axis, len(input_shape), input_shape
            )
        )
    if not axis >= 0:
        if not axis >= -1 * len(input_shape):
            raise ValueError(
                "the axis:{} should not be less than -1 * length of input_shape:{}".format(
                    axis, -1 * len(input_shape)
                )
            )
        axis = axis + len(input_shape)
    if in_dynamic_mode():
        out = _C_ops.renorm(x, p, axis, max_norm)
        return out
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'renorm')
        inputs = {'X': x}
        attrs = {'p': p, 'axis': axis, 'max_norm': max_norm}

        helper = LayerHelper("renorm", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="renorm", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )
        return out


def inner(x, y, name=None):
    """

    Inner product of two input Tensor.

    Ordinary inner product for 1-D Tensors, in higher dimensions a sum product over the last axes.

    Args:
        x (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match y's.
        y (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match x's.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The inner-product Tensor, the output shape is x.shape[:-1] + y.shape[:-1].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
            >>> y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
            >>> out = paddle.inner(x, y)
            >>> print(out)
            >>> #        ([[14, 32, 50],
            >>> #         [32, 77, 122]])


    """
    if x.size == 1 or y.size == 1:
        return multiply(x, y)
    else:
        xshape = x.shape
        yshape = y.shape
        dstshape = list(xshape[:-1]) + list(yshape[:-1])

        nx = x.reshape((-1, xshape[-1]))
        ny = y.reshape((-1, yshape[-1]))

        if in_dynamic_mode():
            return _C_ops.matmul(nx, ny.T, False, False).reshape(dstshape)
        else:

            def __check_input(x, y):
                var_names = {'x': x, 'y': y}
                for name, val in var_names.items():
                    check_variable_and_dtype(
                        val, name, ['float16', 'float32', 'float64'], 'inner'
                    )
                x_shape = list(xshape)
                y_shape = list(yshape)

                # check the inner 2 dimensions
                if x_shape[-1] != y_shape[-1]:
                    if not ((x_shape[-1] == -1) or (y_shape[-1] == -1)):
                        raise ValueError(
                            "After performing an optional transpose, Input X's last dim should be "
                            "equal to Y's last dim for multiplication "
                            "prerequisites. But received X's shape: {}, Y's shape: {}\n".format(
                                x_shape, y_shape
                            )
                        )

            __check_input(nx, ny)

            helper = LayerHelper('inner', **locals())
            out = helper.create_variable_for_type_inference(dtype=nx.dtype)
            helper.append_op(
                type='matmul_v2',
                inputs={'X': nx, 'Y': ny.T},
                outputs={'Out': out},
            )
            return out.reshape(dstshape)


def outer(x, y, name=None):
    """

    Outer product of two Tensors.

    Input is flattened if not already 1-dimensional.

    Args:
        x (Tensor): An N-D Tensor or a Scalar Tensor.
        y (Tensor): An N-D Tensor or a Scalar Tensor.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The outer-product Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.arange(1, 4).astype('float32')
            >>> y = paddle.arange(1, 6).astype('float32')
            >>> out = paddle.outer(x, y)
            >>> print(out)
            >>> #        ([[1, 2, 3, 4, 5],
            >>> #         [2, 4, 6, 8, 10],
            >>> #         [3, 6, 9, 12, 15]])


    """
    nx = x.reshape((-1, 1))
    ny = y.reshape((1, -1))

    if in_dynamic_mode():
        return _C_ops.matmul(nx, ny, False, False)
    else:

        def __check_input(x, y):
            var_names = {'x': x, 'y': y}
            for name, val in var_names.items():
                check_variable_and_dtype(
                    val, name, ['float16', 'float32', 'float64'], 'inner'
                )

        __check_input(nx, ny)

        helper = LayerHelper('outer', **locals())
        out = helper.create_variable_for_type_inference(dtype=nx.dtype)
        helper.append_op(
            type='matmul_v2', inputs={'X': nx, 'Y': ny}, outputs={'Out': out}
        )
        return out


def logsumexp(x, axis=None, keepdim=False, name=None):
    r"""
    Calculates the log of the sum of exponentials of ``x`` along ``axis`` .

    .. math::
       logsumexp(x) = \log\sum exp(x)

    Args:
        x (Tensor): The input Tensor with data type float16, float32 or float64, which
            have no more than 4 dimensions.
        axis (int|list|tuple, optional): The axis along which to perform
            logsumexp calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), logsumexp
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is
            less than 0, it works the same way as :math:`axis + D` . If
            ``axis`` is None, logsumexp is calculated along all elements of
            ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keep_dim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of logsumexp along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:

    .. code-block:: python

        >>> import paddle

        >>> x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
        >>> out1 = paddle.logsumexp(x)    # 3.4691226
        >>> out2 = paddle.logsumexp(x, 1) # [2.15317821, 3.15684602]

    """
    reduce_all, axis = _get_reduce_axis(axis, x)

    if in_dynamic_mode():
        return _C_ops.logsumexp(x, axis, keepdim, reduce_all)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'logsumexp'
        )

        helper = LayerHelper('logsumexp', **locals())
        attrs = {'axis': axis, 'keepdim': keepdim, 'reduce_all': reduce_all}
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logsumexp', inputs={'X': x}, outputs={'Out': out}, attrs=attrs
        )
        return out


def inverse(x, name=None):
    """
    Takes the inverse of the square matrix. A square matrix is a matrix with
    the same number of rows and columns. The input can be a square matrix
    (2-D Tensor) or batches of square matrices.

    Args:
        x (Tensor): The input tensor. The last two
            dimensions should be equal. When the number of dimensions is
            greater than 2, it is treated as batches of square matrix. The data
            type can be float32 and float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor holds the inverse of x. The shape and data type
                        is the same as x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
            >>> inv = paddle.inverse(mat)
            >>> print(inv) # [[0.5, 0], [0, 0.5]]

    """
    if in_dynamic_mode():
        return _C_ops.inverse(x)
    else:

        def _check_input(x):
            check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'inverse')
            if len(x.shape) < 2:
                raise ValueError(
                    "The input of inverse is expected to be a Tensor whose number "
                    "of dimensions is no less than 2. But reviced: %d, "
                    "x's shape: %s." % (len(x.shape), x.shape)
                )

        _check_input(x)
        helper = LayerHelper('inverse', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='inverse', inputs={'Input': [x]}, outputs={'Output': [out]}
        )
        return out


def max(x, axis=None, keepdim=False, name=None):
    """

    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values,
        while max propagates gradient to all of them.


    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                         [0.1, 0.2, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result1 = paddle.max(x)
            >>> result1.backward()
            >>> print(result1, x.grad)
            >>> # 0.9, [[0., 0., 0., 1.], [0., 0., 0., 0.]]

            >>> x.clear_grad()
            >>> result2 = paddle.max(x, axis=0)
            >>> result2.backward()
            >>> print(result2, x.grad)
            >>> #[0.2, 0.3, 0.6, 0.9], [[1., 1., 0., 1.], [0., 0., 1., 0.]]

            >>> x.clear_grad()
            >>> result3 = paddle.max(x, axis=-1)
            >>> result3.backward()
            >>> print(result3, x.grad)
            >>> #[0.9, 0.7], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

            >>> x.clear_grad()
            >>> result4 = paddle.max(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> print(result4, x.grad)
            >>> #[[0.9], [0.7]], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                         [[5.0, 6.0], [7.0, 8.0]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.max(y, axis=[1, 2])
            >>> result5.backward()
            >>> print(result5, y.grad)
            >>> #[4., 8.], [[[0., 0.], [0., 1.]], [[0., 0.], [0., 1.]]]

            >>> y.clear_grad()
            >>> result6 = paddle.max(y, axis=[0, 1])
            >>> result6.backward()
            >>> print(result6, y.grad)
            >>> #[7., 8.], [[[0., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]
    """

    if in_dynamic_mode():
        return _C_ops.max(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        helper = LayerHelper('max', **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'max',
        )
        if not isinstance(axis, Variable) and paddle.utils._contain_var(axis):
            axis = paddle.utils._convert_to_tensor_list(axis)

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_max',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def min(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values,
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # data_x is a Tensor with shape [2, 4]
            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                         [0.1, 0.2, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result1 = paddle.min(x)
            >>> result1.backward()
            >>> print(result1, x.grad)
            >>> # 0.1, [[0., 0., 0., 0.], [1., 0., 0., 0.]]

            >>> x.clear_grad()
            >>> result2 = paddle.min(x, axis=0)
            >>> result2.backward()
            >>> print(result2, x.grad)
            >>> #[0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]

            >>> x.clear_grad()
            >>> result3 = paddle.min(x, axis=-1)
            >>> result3.backward()
            >>> print(result3, x.grad)
            >>> #[0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

            >>> x.clear_grad()
            >>> result4 = paddle.min(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> print(result4, x.grad)
            >>> #[[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                         [[5.0, 6.0], [7.0, 8.0]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.min(y, axis=[1, 2])
            >>> result5.backward()
            >>> print(result5, y.grad)
            >>> #[1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]

            >>> y.clear_grad()
            >>> result6 = paddle.min(y, axis=[0, 1])
            >>> result6.backward()
            >>> print(result6, y.grad)
            >>> #[1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]
    """

    if in_dynamic_mode():
        return _C_ops.min(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        helper = LayerHelper('min', **locals())
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'min',
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_min',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def amax(x, axis=None, keepdim=False, name=None):
    """
    Computes the maximum of tensor elements over the given axis.

    Note:
        The difference between max and amax is: If there are multiple maximum elements,
        amax evenly distributes gradient between these equal values,
        while max propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple, optional): The axis along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of maximum on the specified axis of input tensor,
        it's data type is the same as `x`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple maximum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
            ...                         [0.9, 0.9, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 maximum elements:
            >>> # 1) amax evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while max propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amax(x)
            >>> result1.backward()
            >>> print(result1, x.grad)
            >>> # 0.9, [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]

            >>> x.clear_grad()
            >>> result1_max = paddle.max(x)
            >>> result1_max.backward()
            >>> print(result1_max, x.grad)
            >>> # 0.9, [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]

            >>> ###############################

            >>> x.clear_grad()
            >>> result2 = paddle.amax(x, axis=0)
            >>> result2.backward()
            >>> print(result2, x.grad)
            >>> #[0.9, 0.9, 0.9, 0.9], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]

            >>> x.clear_grad()
            >>> result3 = paddle.amax(x, axis=-1)
            >>> result3.backward()
            >>> print(result3, x.grad)
            >>> #[0.9, 0.9], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]

            >>> x.clear_grad()
            >>> result4 = paddle.amax(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> print(result4, x.grad)
            >>> #[[0.9], [0.9]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
            ...                         [[0.9, 0.9], [0.6, 0.7]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amax(y, axis=[1, 2])
            >>> result5.backward()
            >>> print(result5, y.grad)
            >>> #[0.9., 0.9], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]

            >>> y.clear_grad()
            >>> result6 = paddle.amax(y, axis=[0, 1])
            >>> result6.backward()
            >>> print(result6, y.grad)
            >>> #[0.9., 0.9], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]
    """
    if in_dynamic_mode():
        return _C_ops.amax(x, axis, keepdim)

    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        helper = LayerHelper('amax', **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amax'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_amax',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def amin(x, axis=None, keepdim=False, name=None):
    """

    Computes the minimum of tensor elements over the given axis

    Note:
        The difference between min and amin is: If there are multiple minimum elements,
        amin evenly distributes gradient between these equal values,
        while min propagates gradient to all of them.

    Args:
        x (Tensor): A tensor, the data type is float32, float64, int32, int64,
            the dimension is no more than 4.
        axis (int|list|tuple, optional): The axis along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
            If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the `x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of minimum on the specified axis of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> # data_x is a Tensor with shape [2, 4] with multiple minimum elements
            >>> # the axis is a int element

            >>> x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
            ...                         [0.1, 0.1, 0.6, 0.7]],
            ...                         dtype='float64', stop_gradient=False)
            >>> # There are 5 minimum elements:
            >>> # 1) amin evenly distributes gradient between these equal values,
            >>> #    thus the corresponding gradients are 1/5=0.2;
            >>> # 2) while min propagates gradient to all of them,
            >>> #    thus the corresponding gradient are 1.
            >>> result1 = paddle.amin(x)
            >>> result1.backward()
            >>> print(result1, x.grad)
            >>> # 0.1, [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]

            >>> x.clear_grad()
            >>> result1_min = paddle.min(x)
            >>> result1_min.backward()
            >>> print(result1_min, x.grad)
            >>> # 0.1, [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]

            >>> ###############################

            >>> x.clear_grad()
            >>> result2 = paddle.amin(x, axis=0)
            >>> result2.backward()
            >>> print(result2, x.grad)
            >>> #[0.1, 0.1, 0.1, 0.1], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]

            >>> x.clear_grad()
            >>> result3 = paddle.amin(x, axis=-1)
            >>> result3.backward()
            >>> print(result3, x.grad)
            >>> #[0.1, 0.1], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]

            >>> x.clear_grad()
            >>> result4 = paddle.amin(x, axis=1, keepdim=True)
            >>> result4.backward()
            >>> print(result4, x.grad)
            >>> #[[0.1], [0.1]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]

            >>> # data_y is a Tensor with shape [2, 2, 2]
            >>> # the axis is list
            >>> y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
            ...                         [[0.1, 0.1], [0.6, 0.7]]],
            ...                         dtype='float64', stop_gradient=False)
            >>> result5 = paddle.amin(y, axis=[1, 2])
            >>> result5.backward()
            >>> print(result5, y.grad)
            >>> #[0.1., 0.1], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]

            >>> y.clear_grad()
            >>> result6 = paddle.amin(y, axis=[0, 1])
            >>> result6.backward()
            >>> print(result6, y.grad)
            >>> #[0.1., 0.1], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]
    """
    if in_dynamic_mode():
        return _C_ops.amin(x, axis, keepdim)

    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        helper = LayerHelper('amin', **locals())
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'amin'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='reduce_amin',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def log1p(x, name=None):
    r"""
    Calculates the natural log of the given input tensor, element-wise.

    .. math::
        Out = \ln(x+1)

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the natural log of the input Tensor computed element-wise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[0], [1]], dtype='float32')
            >>> res = paddle.log1p(data)
            >>> # [[0.], [0.6931472]]
    """

    if in_dynamic_mode():
        return _C_ops.log1p(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log1p",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log1p', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
        return out


def log2(x, name=None):
    r"""
    Calculates the log to the base 2 of the given input tensor, element-wise.

    .. math::

        Out = \log_2x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 2 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [2.0]])
            >>> res = paddle.log2(x_i) # [[0.], [1.0]]

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> print(res) # [1.0]

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log2(x_i)
            >>> print(res) # [1.0]
    """
    if in_dynamic_mode():
        return _C_ops.log2(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log2",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log2', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log2", inputs={"X": x}, outputs={"Out": out})
        return out


def log10(x, name=None):
    r"""
    Calculates the log to the base 10 of the given input tensor, element-wise.

    .. math::

        Out = \log_10_x

    Args:
        x (Tensor): Input tensor must be one of the following types: int32, int64, float16, bfloat16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The log to the base 10 of the input Tensor computed element-wise.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> # example 1: x is a float
            >>> x_i = paddle.to_tensor([[1.0], [10.0]])
            >>> res = paddle.log10(x_i) # [[0.], [1.0]]

            >>> # example 2: x is float32
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> print(res) # [1.0]

            >>> # example 3: x is float64
            >>> x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
            >>> paddle.to_tensor(x_i)
            >>> res = paddle.log10(x_i)
            >>> print(res) # [1.0]
    """
    if in_dynamic_mode():
        return _C_ops.log10(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'uint16', 'float32', 'float64'],
            "log10",
        )
        inputs = {'X': [x]}
        helper = LayerHelper('log10', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type="log10", inputs={"X": x}, outputs={"Out": out})
        return out


def clip(x, min=None, max=None, name=None):
    """
    This operator clip all elements in input into the range [ min, max ] and return
    a resulting tensor as the following equation:

    .. math::

        Out = MIN(MAX(x, min), max)

    Args:
        x (Tensor): An N-D Tensor with data type float16, float32, float64, int32 or int64.
        min (float|int|Tensor, optional): The lower bound with type ``float`` , ``int`` or a ``0-D Tensor``
            with shape [] and type ``int32``, ``float16``, ``float32``, ``float64``.
        max (float|int|Tensor, optional): The upper bound with type ``float``, ``int`` or a ``0-D Tensor``
            with shape [] and type ``int32``, ``float16``, ``float32``, ``float64``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor with the same data type and data shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
            >>> out1 = paddle.clip(x1, min=3.5, max=5.0)
            >>> out2 = paddle.clip(x1, min=2.5)
            >>> print(out1)
            >>> # [[3.5, 3.5]
            >>> # [4.5, 5.0]]
            >>> print(out2)
            >>> # [[2.5, 3.5]
            >>> # [[4.5, 6.4]
    """

    x_dtype = str(x.dtype)
    if x_dtype == 'paddle.int32':
        min_ = np.iinfo(np.int32).min
        max_ = np.iinfo(np.int32).max - 2**7
    elif x_dtype == 'paddle.int64':
        min_ = np.iinfo(np.int64).min
        max_ = np.iinfo(np.int64).max - 2**39
    elif x_dtype == 'paddle.float16':
        min_ = float(np.finfo(np.float16).min)
        max_ = float(np.finfo(np.float16).max)
    else:
        min_ = float(np.finfo(np.float32).min)
        max_ = float(np.finfo(np.float32).max)

    if in_dynamic_mode():
        if isinstance(min, Variable):
            min = min.item(0)
        if isinstance(max, Variable):
            max = max.item(0)
        min = min_ if min is None else min
        max = max_ if max is None else max
        return _C_ops.clip(x, min, max)
    else:
        if min is not None:
            check_type(min, 'min', (float, int, Variable), 'clip')
            if isinstance(min, Variable):
                check_dtype(
                    min.dtype,
                    'min',
                    ['float16', 'float32', 'float64', 'int32', 'uint16'],
                    'clip',
                    '(When the type of min in clip is Variable.)',
                )
        if max is not None:
            check_type(max, 'max', (float, int, Variable), 'clip')
            if isinstance(max, Variable):
                check_dtype(
                    max.dtype,
                    'max',
                    ['float16', 'float32', 'float64', 'int32', 'uint16'],
                    'clip',
                    '(When the type of max in clip is Variable.)',
                )

        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            'clip',
        )

        inputs = {'X': x}
        attrs = {'min': min_, 'max': max_}

        if isinstance(min, Variable):
            min.stop_gradient = True
            inputs['Min'] = min
        elif min is not None:
            attrs['min'] = min

        if isinstance(max, Variable):
            max.stop_gradient = True
            inputs['Max'] = max
        elif max is not None:
            attrs['max'] = max

        helper = LayerHelper('clip', **locals())
        output = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype('x')
        )
        helper.append_op(
            type='clip', inputs=inputs, outputs={'Out': [output]}, attrs=attrs
        )

        return output


@inplace_apis_in_dygraph_only
def clip_(x, min=None, max=None, name=None):
    """
    Inplace version of ``clip`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_clip`.
    """
    fmin = float(np.finfo(np.float32).min)
    fmax = float(np.finfo(np.float32).max)
    if isinstance(min, Variable):
        min = min.item(0)
    if isinstance(max, Variable):
        max = max.item(0)
    min = fmin if min is None else min
    max = fmax if max is None else max

    if in_dynamic_mode():
        return _C_ops.clip_(x, min, max)


def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """

    Computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> case1 = paddle.randn([2, 3])
            >>> case2 = paddle.randn([3, 10, 10])
            >>> case3 = paddle.randn([3, 10, 5, 10])
            >>> data1 = paddle.trace(case1) # data1.shape = []
            >>> data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
            >>> data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
    """

    def __check_input(x, offset, axis1, axis2):
        check_dtype(
            x.dtype,
            'Input',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'trace',
        )

        input_shape = list(x.shape)
        assert len(input_shape) >= 2, (
            "The x must be at least 2-dimensional, "
            "But received Input x's dimensional: %s.\n" % len(input_shape)
        )

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert (0 <= axis1_) and (axis1_ < len(input_shape)), (
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape)), len(input_shape) - 1, axis1)
        )

        assert (0 <= axis2_) and (axis2_ < len(input_shape)), (
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"
            % (-(len(input_shape)), len(input_shape) - 1, axis2)
        )

        assert axis1_ != axis2_, (
            "axis1 and axis2 cannot be the same axis."
            "But received axis1 = %d, axis2 = %d\n" % (axis1, axis2)
        )

    if in_dynamic_mode():
        return _C_ops.trace(x, offset, axis1, axis2)
    else:
        __check_input(x, offset, axis1, axis2)

        helper = LayerHelper('trace', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='trace',
            inputs={'Input': [x]},
            attrs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            outputs={'Out': [out]},
        )
        return out


def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    """
    Computes the diagonals of the input tensor x.

    If ``x`` is 2D, returns the diagonal.
    If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2.
    By default, the 2D planes formed by the first and second axis of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        x (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32, int64, float16, float32, float64.
        offset (int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1 (int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2 (int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([2,2,3],'float32')
            >>> print(x)
            >>> # Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [[[0.45661032, 0.03751532, 0.90191704],
            >>> #          [0.43760979, 0.86177313, 0.65221709]],

            >>> #         [[0.17020577, 0.00259554, 0.28954273],
            >>> #          [0.51795638, 0.27325270, 0.18117726]]])

            >>> out1 = paddle.diagonal(x)
            >>> print(out1)
            >>> #Tensor(shape=[3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[0.45661032, 0.51795638],
            >>> #        [0.03751532, 0.27325270],
            >>> #        [0.90191704, 0.18117726]])

            >>> out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
            >>> print(out2)
            >>> #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[0.45661032, 0.86177313],
            >>> #        [0.17020577, 0.27325270]])

            >>> out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
            >>> print(out3)
            >>> #Tensor(shape=[3, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[0.43760979],
            >>> #        [0.86177313],
            >>> #        [0.65221709]])

            >>> out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
            >>> print(out4)
            >>> #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[0.45661032, 0.86177313],
            >>> #        [0.17020577, 0.27325270]])

    """
    if in_dynamic_mode():
        return _C_ops.diagonal(x, offset, axis1, axis2)
    else:

        def __check_input(x, offset, axis1, axis2):
            check_dtype(
                x.dtype,
                'Input',
                [
                    'bool',
                    'int32',
                    'int64',
                    'float16',
                    'uint16',
                    'float32',
                    'float64',
                ],
                'diagonal',
            )

            input_shape = list(x.shape)
            assert len(input_shape) >= 2, (
                "The x must be at least 2-dimensional, "
                "But received Input x's dimensional: %s.\n" % len(input_shape)
            )

            axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
            axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

            assert axis1_ < len(input_shape), (
                "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"
                % (-(len(input_shape)), len(input_shape) - 1, axis1)
            )

            assert axis2_ < len(input_shape), (
                "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"
                % (-(len(input_shape)), len(input_shape) - 1, axis2)
            )

            assert axis1_ != axis2_, (
                "axis1 and axis2 cannot be the same axis."
                "But received axis1 = %d, axis2 = %d\n" % (axis1, axis2)
            )

        __check_input(x, offset, axis1, axis2)
        helper = LayerHelper('diagonal', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='diagonal',
            inputs={'Input': [x]},
            attrs={'offset': offset, 'axis1': axis1, 'axis2': axis2},
            outputs={'Out': [out]},
        )
        return out


def kron(x, y, name=None):
    r"""
    Compute the Kronecker product of two tensors, a
    composite tensor made of blocks of the second tensor scaled by the
    first.
    Assume that the rank of the two tensors, $X$ and $Y$
    are the same, if necessary prepending the smallest with ones. If the
    shape of $X$ is [$r_0$, $r_1$, ..., $r_N$] and the shape of $Y$ is
    [$s_0$, $s_1$, ..., $s_N$], then the shape of the output tensor is
    [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. The elements are
    products of elements from $X$ and $Y$.
    The equation is:
    $$
    output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
    Y[j_{0}, j_{1}, ..., j_{N}]
    $$
    where
    $$
    k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N
    $$

    Args:
        x (Tensor): the fist operand of kron op, data type: float16, float32, float64, int32 or int64.
        y (Tensor): the second operand of kron op, data type: float16, float32, float64, int32 or int64. Its data type should be the same with x.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output of kron, data type: float16, float32, float64, int32 or int64. Its data is the same with x.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
            >>> y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
            >>> out = paddle.kron(x, y)
            >>> print(out)
            >>> #        [[1, 2, 3, 2, 4, 6],
            >>> #         [ 4,  5,  6,  8, 10, 12],
            >>> #         [ 7,  8,  9, 14, 16, 18],
            >>> #         [ 3,  6,  9,  4,  8, 12],
            >>> #         [12, 15, 18, 16, 20, 24],
            >>> #         [21, 24, 27, 28, 32, 36]])
    """
    if in_dynamic_mode():
        return _legacy_C_ops.kron(x, y)
    else:
        helper = LayerHelper('kron', **locals())
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron'
        )
        check_variable_and_dtype(
            y, 'y', ['float16', 'float32', 'float64', 'int32', 'int64'], 'kron'
        )

        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="kron", inputs={"X": x, "Y": y}, outputs={"Out": out}
        )
        return out


def cumsum(x, axis=None, dtype=None, name=None):
    """
    The cumulative sum of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cumsumed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float16, float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumsum operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12)
            >>> data = paddle.reshape(data, (3, 4))

            >>> y = paddle.cumsum(data)
            >>> # [ 0  1  3  6 10 15 21 28 36 45 55 66]

            >>> y = paddle.cumsum(data, axis=0)
            >>> # [[ 0  1  2  3]
            >>> #  [ 4  6  8 10]
            >>> #  [12 15 18 21]]

            >>> y = paddle.cumsum(data, axis=-1)
            >>> # [[ 0  1  3  6]
            >>> #  [ 4  9 15 22]
            >>> #  [ 8 17 27 38]]

            >>> y = paddle.cumsum(data, dtype='float64')
            >>> print(y.dtype)
            >>> # paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        if axis is None:
            axis = -1
        return _C_ops.cumsum(x, axis, flatten, False, False)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
            'cumsum',
        )
        check_type(x, 'x', (Variable), 'cumsum')
        locals_var = locals().copy()
        kwargs = {}
        for name, val in locals_var.items():
            if val is not None:
                kwargs[name] = val
        _cum_sum_ = generate_layer_fn('cumsum')
        return _cum_sum_(**kwargs)


def cummax(x, axis=None, dtype='int64', name=None):
    """
    The cumulative max of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cummaxed.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cummax over the flattened array.
        dtype (str, optional): The data type of the indices tensor, can be int32, int64. The default value is int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), The result of cummax operation. The dtype of cummax result is same with input x.

        indices (Tensor), The corresponding index results of cummax operation.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([-1, 5, 0, -2, -3, 2])
            >>> data = paddle.reshape(data, (2, 3))

            >>> y = paddle.cummax(data)
            >>> # value: [-1, 5, 5, 5, 5, 5]
            >>> # indcies: [0, 1, 1, 1, 1, 1]

            >>> y = paddle.cummax(data, axis=0)
            >>> # value: [[-1, 5, 0]
            >>> #         [-1, 5, 2]]
            >>> # indcies: [[0, 0, 0]
            >>> #           [0, 0, 1]]

            >>> y = paddle.cummax(data, axis=-1)
            >>> # value: [[-1, 5, 5]
            >>> #         [-2, -2, 2]]
            >>> # indcies: [[0, 1, 1]
            >>> #           [0, 0, 2]]

            >>> y = paddle.cummax(data, dtype='int64')
            >>> print(y[1].dtype)
            >>> # indcies type: paddle.int64
    """
    if axis is None:
        axis = -1
        x = x.flatten(0, len(x.shape) - 1)

    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'cummax')
    dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.cummax(x, axis, dtype)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float64', 'int32', 'int64'],
            'cummax',
        )
        check_type(x, 'x', (Variable), 'cummax')
        helper = LayerHelper('cummax', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        indices = helper.create_variable_for_type_inference(dtype='int64')
        helper.append_op(
            type='cummax',
            inputs={'x': x},
            outputs={'out': out, 'indices': indices},
            attrs={'axis': axis, 'dtype': dtype},
        )
        return out, indices


def cummin(x, axis=None, dtype='int64', name=None):
    """
    The cumulative min of the elements along a given axis.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor needed to be cummined.
        axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cummin over the flattened array.
        dtype (str, optional): The data type of the indices tensor, can be int32, int64. The default value is int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), The result of cummin operation. The dtype of cummin result is same with input x.

        indices (Tensor), The corresponding index results of cummin operation.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> data = paddle.to_tensor([-1, 5, 0, -2, -3, 2])
            >>> data = paddle.reshape(data, (2, 3))

            >>> y = paddle.cummin(data)
            >>> # value: [-1, -1, -1, -2, -3, -3]
            >>> # indcies: [0, 0, 0, 3, 4, 4]

            >>> y = paddle.cummin(data, axis=0)
            >>> # value: [[-1, 5, 0]
            >>> #         [-2, -3, 0]]
            >>> # indcies: [[0, 0, 0]
            >>> #           [1, 1, 0]]

            >>> y = paddle.cummin(data, axis=-1)
            >>> # value: [[-1, -1, -1]
            >>> #         [-2, -3, -3]]
            >>> # indcies: [[0, 0, 0]
            >>> #           [0, 1, 1]]

            >>> y = paddle.cummin(data, dtype='int64')
            >>> print(y[1].dtype)
            >>> # indcies type: paddle.int64
    """
    if axis is None:
        axis = -1
        x = x.flatten(0, len(x.shape) - 1)

    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'cummin')
    dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dynamic_mode():
        return _C_ops.cummin(x, axis, dtype)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float32', 'float64', 'int32', 'int64'],
            'cummin',
        )
        check_type(x, 'x', (Variable), 'cummin')
        helper = LayerHelper('cummin', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        indices = helper.create_variable_for_type_inference(dtype='int64')
        helper.append_op(
            type='cummin',
            inputs={'x': x},
            outputs={'out': out, 'indices': indices},
            attrs={'axis': axis, 'dtype': dtype},
        )
        return out, indices


def logcumsumexp(x, axis=None, dtype=None, name=None):
    r"""
    The logarithm of the cumulative summation of the exponentiation of the elements along a given axis.

    For summation index j given by `axis` and other indices i, the result is

    .. math::

        logcumsumexp(x)_{ij} = log \sum_{i=0}^{j}exp(x_{ij})

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): The input tensor.
        axis (int, optional): The dimension to do the operation along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
        dtype (str, optional): The data type of the output tensor, can be float16, float32, float64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of logcumsumexp operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12, dtype='float64')
            >>> data = paddle.reshape(data, (3, 4))

            >>> y = paddle.logcumsumexp(data)
            >>> # [ 0.         1.3132617  2.4076061  3.4401898  4.4519143  5.4561934
            >>> #   6.4577627  7.4583397  8.458551   9.45863   10.458658  11.458669 ]

            >>> y = paddle.logcumsumexp(data, axis=0)
            >>> # [[ 0.        1.        2.        3.      ]
            >>> #  [ 4.01815   5.01815   6.01815   7.01815 ]
            >>> #  [ 8.018479  9.018479 10.018479 11.018479]]

            >>> y = paddle.logcumsumexp(data, axis=-1)
            >>> # [[ 0.         1.3132617  2.4076061  3.4401898]
            >>> #  [ 4.         5.3132615  6.407606   7.44019  ]
            >>> #  [ 8.         9.313262  10.407606  11.440189 ]]

            >>> y = paddle.logcumsumexp(data, dtype='float64')
            >>> print(y.dtype)
            >>> # paddle.float64
    """
    if axis is None:
        flatten = True
    else:
        flatten = False
    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        if axis is None:
            axis = -1
        return _C_ops.logcumsumexp(x, axis, flatten, False, False)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], "logcumsumexp"
        )

        helper = LayerHelper('logcumsumexp', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logcumsumexp',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'axis': axis, 'flatten': flatten},
        )
        return out


def cumprod(x, dim=None, dtype=None, name=None):
    """
    Compute the cumulative product of the input tensor x along a given dimension dim.

    Note:
        The first element of the result is the same as the first element of the input.

    Args:
        x (Tensor): the input tensor need to be cumproded.
        dim (int, optional): the dimension along which the input tensor will be accumulated. It need to be in the range of [-x.rank, x.rank),
                    where x.rank means the dimensions of the input tensor x and -1 means the last dimension.
        dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64, complex64,
                    complex128. If specified, the input tensor is casted to dtype before the operation is performed.
                    This is useful for preventing data type overflows. The default value is None.
        name (str, optional): Name for the operation (optional, default is None). For more information,
                    please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the result of cumprod operator.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.arange(12)
            >>> data = paddle.reshape(data, (3, 4))
            >>> # [[ 0  1  2  3 ]
            >>> #  [ 4  5  6  7 ]
            >>> #  [ 8  9  10 11]]

            >>> y = paddle.cumprod(data, dim=0)
            >>> # [[ 0  1   2   3]
            >>> #  [ 0  5  12  21]
            >>> #  [ 0 45 120 231]]

            >>> y = paddle.cumprod(data, dim=-1)
            >>> # [[ 0   0   0    0]
            >>> #  [ 4  20 120  840]
            >>> #  [ 8  72 720 7920]]

            >>> y = paddle.cumprod(data, dim=1, dtype='float64')
            >>> # [[ 0.   0.   0.    0.]
            >>> #  [ 4.  20. 120.  840.]
            >>> #  [ 8.  72. 720. 7920.]]

            >>> print(y.dtype)
            >>> # paddle.float64

    """

    if dtype is not None and x.dtype != convert_np_dtype_to_dtype_(dtype):
        x = cast(x, dtype)

    if in_dynamic_mode():
        return _C_ops.cumprod(x, dim)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'complex64',
                'complex128',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            'cumprod',
        )
        check_type(dim, 'dim', int, 'cumprod')

        helper = LayerHelper('cumprod', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='cumprod',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': dim},
        )
        return out


def isfinite(x, name=None):
    """

    Return whether every element of input tensor is finite number or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isfinite(x)
            >>> print(out)  # [False  True  True False  True False False]
    """
    if in_dynamic_mode():
        return _C_ops.isfinite(x)
    else:
        helper = LayerHelper("isfinite_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isfinite',
        )
        out = helper.create_variable_for_type_inference('bool')
        helper.append_op(
            type="isfinite_v2", inputs={"X": x}, outputs={"Out": out}
        )
        return out


def isinf(x, name=None):
    """

    Return whether every element of input tensor is `+/-INF` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isinf(x)
            >>> print(out)  # [ True False False  True False False False]
    """
    if in_dynamic_mode():
        return _C_ops.isinf(x)
    else:
        helper = LayerHelper("isinf_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isinf',
        )
        out = helper.create_variable_for_type_inference(dtype='bool')
        helper.append_op(type="isinf_v2", inputs={"X": x}, outputs={"Out": out})
        return out


def isnan(x, name=None):
    """

    Return whether every element of input tensor is `NaN` or not.

    Args:
        x (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
            >>> out = paddle.isnan(x)
            >>> print(out)  # [False False False False False  True  True]
    """
    if in_dynamic_mode():
        return _C_ops.isnan(x)
    else:
        helper = LayerHelper("isnan_v2", **locals())
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'uint16',
            ],
            'isnan',
        )
        out = helper.create_variable_for_type_inference(dtype='bool')
        helper.append_op(type="isnan_v2", inputs={"X": x}, outputs={"Out": out})
        return out


def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    """
    Compute the product of tensor elements over the given axis.

    Args:
        x (Tensor): The input tensor, its data type should be float32, float64, int32, int64.
        axis (int|list|tuple, optional): The axis along which the product is computed. If :attr:`None`,
            multiply all elements of `x` and return a Tensor with a single element,
            otherwise must be in the range :math:`[-x.ndim, x.ndim)`. If :math:`axis[i]<0`,
            the axis to reduce is :math:`x.ndim + axis[i]`. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result
            tensor will have one fewer dimension than the input unless `keepdim` is true. Default is False.
        dtype (str|np.dtype, optional): The desired date type of returned tensor, can be float32, float64,
            int32, int64. If specified, the input tensor is casted to dtype before operator performed.
            This is very useful for avoiding data type overflows. The default value is None, the dtype
            of output is the same as input Tensor `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, result of product on the specified dim of input tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # the axis is a int element
            >>> x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
            ...                         [0.1, 0.2, 0.6, 0.7]])
            >>> out1 = paddle.prod(x)
            >>> # 0.0002268

            >>> out2 = paddle.prod(x, -1)
            >>> # [0.027  0.0084]

            >>> out3 = paddle.prod(x, 0)
            >>> # [0.02 0.06 0.3  0.63]

            >>> out4 = paddle.prod(x, 0, keepdim=True)
            >>> # [[0.02 0.06 0.3  0.63]]

            >>> out5 = paddle.prod(x, 0, dtype='int64')
            >>> # [0 0 0 0]

            >>> # the axis is list
            >>> y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
            ...                         [[5.0, 6.0], [7.0, 8.0]]])
            >>> out6 = paddle.prod(y, [0, 1])
            >>> # [105. 384.]

            >>> out7 = paddle.prod(y, (1, 2))
            >>> # [  24. 1680.]

    """
    if dtype is not None:
        check_dtype(
            dtype,
            'dtype',
            ['float32', 'float64', 'int32', 'int64', "float16", "uint16"],
            'prod',
        )
        if x.dtype != convert_np_dtype_to_dtype_(dtype):
            x = cast(x, dtype)

    reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
    if in_dynamic_mode():
        return _C_ops.prod(x, axis, keepdim, reduce_all)
    else:
        helper = LayerHelper('reduce_prod', **locals())
        check_variable_and_dtype(
            x,
            'x/input',
            ['float32', 'float64', 'int32', 'int64', "float16", "uint16"],
            'reduce_prod',
        )
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        helper.append_op(
            type='reduce_prod',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all},
        )
        return out


def sign(x, name=None):
    """
    Returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x (Tensor): The input tensor. The data type can be float16, float32 or float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
            >>> out = paddle.sign(x=x)
            >>> print(out)  # [1.0, 0.0, -1.0, 1.0]
    """
    if in_dynamic_mode():
        return _C_ops.sign(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'sign'
        )
        helper = LayerHelper("sign", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

        return out


def tanh(x, name=None):
    r"""
    Tanh Activation Operator.

    .. math::
        out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Args:
        x (Tensor): Input of Tanh operator, an N-D Tensor, with data type bfloat16, float32, float64 or float16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output of Tanh operator, a Tensor with same data type and shape as input.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.tanh(x)
            >>> print(out)
            >>> # [-0.37994896 -0.19737532  0.09966799  0.29131261]
    """
    if in_dynamic_mode():
        return _C_ops.tanh(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'tanh'
        )
        check_type(x, 'x', (Variable), 'tanh')
        helper = LayerHelper('tanh', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='tanh', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def tanh_(x, name=None):
    r"""
    Inplace version of ``tanh`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_tanh`.
    """
    return _C_ops.tanh_(x)


def increment(x, value=1.0, name=None):
    """
    The API is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
    Notice that the number of elements in :attr:`x` must be equal to 1.

    Args:
        x (Tensor): A tensor that must always contain only one element, its data type supports float32, float64, int32 and int64.
        value (float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the elementwise-incremented tensor with the same shape and data type as :attr:`x`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.zeros(shape=[1], dtype='float32')
            >>> counter = paddle.increment(data)
            >>> # [1.]

    """
    if in_dynamic_mode():
        return _C_ops.increment_(x, value)
    else:
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64', 'int32', 'int64'], 'increment'
        )
        helper = LayerHelper("increment", **locals())
        helper.append_op(
            type='increment',
            inputs={'X': [x]},
            outputs={'Out': [x]},
            attrs={'step': float(value)},
        )
        return x


def all(x, axis=None, keepdim=False, name=None):
    """
    Computes the ``logical and`` of tensor elements over the given dimension.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical and`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results the ``logical and`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]
            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> print(x)
            >>> x = paddle.cast(x, 'bool')

            >>> # out1 should be False
            >>> out1 = paddle.all(x)          # False
            >>> print(out1)

            >>> # out2 should be [True, False]
            >>> out2 = paddle.all(x, axis=0)  # [True, False]
            >>> print(out2)

            >>> # keepdim=False, out3 should be [False, True], out.shape should be (2,)
            >>> out3 = paddle.all(x, axis=-1) # [False, True]
            >>> print(out3)

            >>> # keepdim=True, out4 should be [[False], [True]], out.shape should be (2,1)
            >>> out4 = paddle.all(x, axis=1, keepdim=True) # [[False], [True]]
            >>> print(out4)

    """
    if in_dynamic_mode():
        return _C_ops.all(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        attrs = {
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all,
        }
        check_variable_and_dtype(
            x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'all'
        )
        check_type(axis, 'axis', (int, list, tuple, type(None)), 'all')

        helper = LayerHelper('all', **locals())
        out = helper.create_variable_for_type_inference(dtype=paddle.bool)
        helper.append_op(
            type='reduce_all',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def any(x, axis=None, keepdim=False, name=None):
    """
    Computes the ``logical or`` of tensor elements over the given dimension, and return the result.

    Args:
        x (Tensor): An N-D Tensor, the input data type should be `bool`.
        axis (int|list|tuple, optional): The dimensions along which the ``logical or`` is compute. If
            :attr:`None`, and all elements of :attr:`x` and return a
            Tensor with a single element, otherwise must be in the
            range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
            the dimension to reduce is :math:`rank + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result Tensor will have one fewer dimension
            than the :attr:`x` unless :attr:`keepdim` is true, default
            value is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: Results the ``logical or`` on the specified axis of input Tensor `x`,  it's data type is bool.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
            >>> x = paddle.assign(x)
            >>> print(x)
            >>> x = paddle.cast(x, 'bool')
            >>> # x is a bool Tensor with following elements:
            >>> #    [[True, False]
            >>> #     [True, True]]

            >>> # out1 should be True
            >>> out1 = paddle.any(x)           # True
            >>> print(out1)

            >>> # out2 should be [True, True]
            >>> out2 = paddle.any(x, axis=0)   # [True, True]
            >>> print(out2)

            >>> # keepdim=False, out3 should be [True, True], out.shape should be (2,)
            >>> out3 = paddle.any(x, axis=-1)  # [True, True]
            >>> print(out3)

            >>> # keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
            >>> out4 = paddle.any(x, axis=1, keepdim=True)  # [[True], [True]]
            >>> print(out4)

    """
    if in_dynamic_mode():
        return _C_ops.any(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis(axis, x)
        attrs = {
            'dim': axis,
            'keep_dim': keepdim,
            'reduce_all': reduce_all,
        }
        check_variable_and_dtype(
            x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'any'
        )
        check_type(axis, 'axis', (int, list, tuple, type(None)), 'any')

        helper = LayerHelper('any', **locals())
        out = helper.create_variable_for_type_inference(dtype=paddle.bool)
        helper.append_op(
            type='reduce_any',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def broadcast_shape(x_shape, y_shape):
    """
    The function returns the shape of doing operation with broadcasting on tensors of x_shape and y_shape.

    Note:
        If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x_shape (list[int]|tuple[int]): A shape of tensor.
        y_shape (list[int]|tuple[int]): A shape of tensor.


    Returns:
        list[int], the result shape.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
            >>> # [2, 3, 3]

            >>> # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
            >>> # ValueError (terminated with error message).

    """

    return core.broadcast_shape(x_shape, y_shape)


def conj(x, name=None):
    r"""
    This function computes the conjugate of the Tensor elementwisely.

    Args:
        x (Tensor): The input Tensor which hold the complex numbers.
            Optional data types are:float16, complex64, complex128, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The conjugate of input. The shape and data type is the same with input. If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
            >>> #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[(1+1j), (2+2j), (3+3j)],
            >>> #        [(4+4j), (5+5j), (6+6j)]])

            >>> conj_data=paddle.conj(data)
            >>> #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[(1-1j), (2-2j), (3-3j)],
            >>> #        [(4-4j), (5-5j), (6-6j)]])

    """
    if in_dynamic_mode():
        return _C_ops.conj(x)
    else:
        check_variable_and_dtype(
            x,
            "x",
            [
                'complex64',
                'complex128',
                'float16',
                'uint16',
                'float32',
                'float64',
                'int32',
                'int64',
            ],
            'conj',
        )

        helper = LayerHelper('conj', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )

        helper.append_op(type='conj', inputs={'X': x}, outputs={'Out': [out]})
        return out


def digamma(x, name=None):
    r"""
    Calculates the digamma of the given input tensor, element-wise.

    .. math::
        Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the digamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
            >>> res = paddle.digamma(data)
            >>> print(res)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [[-0.57721591,  0.03648996],
            >>> #        [ nan       ,  5.32286835]])
    """

    if in_dynamic_mode():
        return _C_ops.digamma(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'digamma'
        )
        helper = LayerHelper('digamma', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='digamma', inputs={'X': x}, outputs={'Out': out})
        return out


def lgamma(x, name=None):
    r"""
    Calculates the lgamma of the given input tensor, element-wise.

    This operator performs elementwise lgamma for input $X$.
    :math:`out = log\Gamma(x)`


    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float16, float32, float64, uint16.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the lgamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.lgamma(x)
            >>> print(out)
            >>> # [1.31452441, 1.76149750, 2.25271273, 1.09579802]
    """
    if in_dynamic_mode():
        return _C_ops.lgamma(x)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'lgamma'
        )
        helper = LayerHelper('lgamma', **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(type='lgamma', inputs={'X': x}, outputs={'Out': out})
        return out


def neg(x, name=None):
    """
    This function computes the negative of the Tensor elementwisely.

    Args:
        x (Tensor): Input of neg operator, an N-D Tensor, with data type float32, float64, int8, int16, int32, or int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The negative of input Tensor. The shape and data type are the same with input Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            >>> out = paddle.neg(x)
            >>> print(out)
            >>> # [0.4 0.2 -0.1 -0.3]
    """

    return scale(
        x, scale=-1.0, bias=0.0, bias_after_scale=True, act=None, name=name
    )


def atan2(x, y, name=None):
    r"""
    Element-wise arctangent of x/y with consideration of the quadrant.

    Equation:
        .. math::

            atan2(x,y)=\left\{\begin{matrix}
            & tan^{-1}(\frac{x}{y}) & y > 0 \\
            & tan^{-1}(\frac{x}{y}) + \pi & x>=0, y < 0 \\
            & tan^{-1}(\frac{x}{y}) - \pi & x<0, y < 0 \\
            & +\frac{\pi}{2} & x>0, y = 0 \\
            & -\frac{\pi}{2} & x<0, y = 0 \\
            &\text{undefined} & x=0, y = 0
            \end{matrix}\right.

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64, float16, float32, float64.
        y (Tensor): An N-D Tensor, must have the same type as `x`.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float64 when the input data type is int).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
            >>> #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [-1,  1,  1, -1])

            >>> y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
            >>> #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [-1,  -1,  1, 1])

            >>> out = paddle.atan2(x, y)
            >>> #Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [-2.35619450,  2.35619450,  0.78539819, -0.78539819])

    """

    if in_dynamic_mode():
        return _C_ops.atan2(x, y)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'atan2',
        )
        check_variable_and_dtype(
            y,
            'y',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'atan2',
        )

        helper = LayerHelper('atan2', **locals())
        inputs = {'X1': x, 'X2': y}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='atan2', inputs=inputs, outputs={'Out': out})
        return out


def logit(x, eps=None, name=None):
    r"""
    This function generates a new tensor with the logit of the elements of input x. x is clamped to [eps, 1-eps] when eps is not zero. When eps is zero and x < 0 or x > 1, the function will yields NaN.

    .. math::

        logit(x) = ln(\frac{x}{1 - x})

    where

    .. math::

        x_i=
            \left\{\begin{array}{rcl}
                x_i & &\text{if } eps == Default \\
                eps & &\text{if } x_i < eps \\
                x_i & &\text{if } eps <= x_i <= 1-eps \\
                1-eps & &\text{if } x_i > 1-eps
            \end{array}\right.

    Args:
        x (Tensor): The input Tensor with data type bfloat16, float16, float32, float64.
        eps (float, optional):  the epsilon for input clamp bound. Default is None.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out(Tensor): A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0.2635, 0.0106, 0.2780, 0.2097, 0.8095])
            >>> out1 = paddle.logit(x)
            >>> print(out1)
            >>> # [-1.0277, -4.5365, -0.9544, -1.3269,  1.4468]

    """
    if eps is None:
        eps = 0.0
    if in_dynamic_mode():
        return _C_ops.logit(x, eps)
    else:
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'logit'
        )
        helper = LayerHelper("logit", **locals())
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='logit',
            inputs={'X': x},
            outputs={'Out': out},
            attrs={'eps': eps},
        )
        return out


def lerp(x, y, weight, name=None):
    r"""
    Does a linear interpolation between x and y based on weight.

    Equation:
        .. math::

            lerp(x, y, weight) = x + weight * (y - x).

    Args:
        x (Tensor): An N-D Tensor with starting points, the data type is bfloat16, float16, float32, float64.
        y (Tensor): An N-D Tensor with ending points, the data type is bfloat16, float16, float32, float64.
        weight (float|Tensor): The weight for the interpolation formula. When weight is Tensor, the data type is bfloat16, float16, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(1., 5., dtype='float32')
            >>> y = paddle.empty([4], dtype='float32')
            >>> y.fill_(10.)
            >>> out = paddle.lerp(x, y, 0.5)
            >>> # out: [5.5, 6., 6.5, 7.]

    """
    if isinstance(weight, float):
        weight = paddle.full(shape=[], fill_value=weight, dtype=x.dtype)

    if in_dynamic_mode():
        return _C_ops.lerp(x, y, weight)
    else:
        check_variable_and_dtype(
            x, 'x', ['uint16', 'float16', 'float32', 'float64'], 'lerp'
        )
        check_variable_and_dtype(
            y, 'y', ['uint16', 'float16', 'float32', 'float64'], 'lerp'
        )
        check_variable_and_dtype(
            weight,
            'weight',
            ['uint16', 'float16', 'float32', 'float64'],
            'lerp',
        )

        helper = LayerHelper('lerp', **locals())
        inputs = {'X': x, 'Y': y, 'Weight': weight}
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='lerp', inputs=inputs, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def lerp_(x, y, weight, name=None):
    r"""
    Inplace version of ``lerp`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_lerp`.
    """
    out_shape = broadcast_shape(x.shape, y.shape)
    check_type(weight, 'weight', (float, paddle.Tensor, Variable), 'lerp')
    if isinstance(weight, float):
        weight = paddle.to_tensor([weight], dtype=x.dtype)
    elif isinstance(weight, (paddle.Tensor, Variable)):
        out_shape = broadcast_shape(out_shape, weight.shape)
    if out_shape != x.shape:
        raise ValueError(
            "The shape of broadcast output {} is different from that of inplace tensor {} in the Inplace operation.".format(
                out_shape, x.shape
            )
        )
    return _C_ops.lerp_(x, y, weight)


def erfinv(x, name=None):
    r"""
    The inverse error function of x. Please refer to :ref:`api_paddle_erf`

        .. math::

            erfinv(erf(x)) = x.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor), an N-D Tensor, the shape and data type is the same with input.

    Example:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
            >>> out = paddle.erfinv(x)
            >>> # out: [0, 0.4769, -inf]

    """
    if in_dynamic_mode():
        return _C_ops.erfinv(x)
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'erfinv')
        helper = LayerHelper('erfinv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='erfinv', inputs={'X': x}, outputs={'Out': out})
        return out


@inplace_apis_in_dygraph_only
def erfinv_(x, name=None):
    r"""
    Inplace version of ``erfinv`` API, the output Tensor will be inplaced with input ``x``.
    Please refer to :ref:`api_tensor_erfinv`.
    """
    check_type(x, 'x', (paddle.Tensor, Variable), 'erfinv')
    return _C_ops.erfinv_(x)


def rad2deg(x, name=None):
    r"""
    Convert each of the elements of input x from angles in radians to degrees.

    Equation:
        .. math::

            rad2deg(x)=180/ \pi * x

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import math

            >>> x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
            >>> result1 = paddle.rad2deg(x1)
            >>> print(result1)
            >>> # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #         [180.02334595, -180.02334595,  359.98937988, -359.98937988,
            >>> #           9.95437622 , -89.95437622])

            >>> x2 = paddle.to_tensor(math.pi/2)
            >>> result2 = paddle.rad2deg(x2)
            >>> print(result2)
            >>> # Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #         90.)

            >>> x3 = paddle.to_tensor(1)
            >>> result3 = paddle.rad2deg(x3)
            >>> print(result3)
            >>> # Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        57.29578018)
    """
    rad2deg_scale = 180 / np.pi
    if in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, rad2deg_scale, 0.0, True)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float32', 'float64'], 'rad2deg'
        )
        helper = LayerHelper('rad2deg', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': out_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': paddle.float32},
            )
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale',
            inputs={'X': out_cast},
            outputs={'Out': out},
            attrs={'scale': rad2deg_scale},
        )
        return out


def deg2rad(x, name=None):
    r"""
    Convert each of the elements of input x from degrees to angles in radians.

        .. math::

            deg2rad(x)=\pi * x / 180

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
            >>> result1 = paddle.deg2rad(x1)
            >>> print(result1)
            >>> # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #         [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
            >>> #           -1.57079637])

            >>> x2 = paddle.to_tensor(180)
            >>> result2 = paddle.deg2rad(x2)
            >>> print(result2)
            >>> # Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        3.14159274)
    """
    deg2rad_scale = np.pi / 180.0
    if in_dynamic_mode():
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            x = cast(x, dtype="float32")
        return _C_ops.scale(x, deg2rad_scale, 0.0, True)
    else:
        check_variable_and_dtype(
            x, 'x', ['int32', 'int64', 'float32', 'float64'], 'deg2rad'
        )
        helper = LayerHelper('deg2rad', **locals())
        out_cast = x
        if convert_dtype(x.dtype) in ['int32', 'int64']:
            out_cast = helper.create_variable_for_type_inference(
                dtype=paddle.float32
            )
            helper.append_op(
                type='cast',
                inputs={'X': x},
                outputs={'Out': out_cast},
                attrs={'in_dtype': x.dtype, 'out_dtype': paddle.float32},
            )
        out = helper.create_variable_for_type_inference(dtype=out_cast.dtype)
        helper.append_op(
            type='scale',
            inputs={'X': out_cast},
            outputs={'Out': out},
            attrs={'scale': deg2rad_scale},
        )
        return out


def gcd(x, y, name=None):
    """
    Computes the element-wise greatest common divisor (GCD) of input |x| and |y|.
    Both x and y must have integer types.

    Note:
        gcd(0,0)=0, gcd(0, y)=|y|

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64.
        y (Tensor): An N-D Tensor, the data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor(12)
            >>> x2 = paddle.to_tensor(20)
            >>> paddle.gcd(x1, x2)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        4)

            >>> x3 = paddle.arange(6)
            >>> paddle.gcd(x3, x2)
            >>> # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [20, 1 , 2 , 1 , 4 , 5])

            >>> x4 = paddle.to_tensor(0)
            >>> paddle.gcd(x4, x2)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        20)

            >>> paddle.gcd(x4, x4)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        0)

            >>> x5 = paddle.to_tensor(-20)
            >>> paddle.gcd(x1, x5)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        4)
    """
    shape = paddle.broadcast_shape(x.shape, y.shape)
    x = paddle.broadcast_to(x, shape)
    y = paddle.broadcast_to(y, shape)
    x = paddle.abs(x)
    y = paddle.abs(y)

    def _gcd_cond_fn(x, y):
        return paddle.any(y != 0)

    def _gcd_body_fn(x, y):
        # paddle.mod will raise an error when any element of y is 0. To avoid
        # that, we change those zeros to ones. Their values don't matter because
        # they won't be used.
        y_not_equal_0 = y != 0
        y_safe = paddle.where(y_not_equal_0, y, paddle.ones(y.shape, y.dtype))
        x, y = (
            paddle.where(y_not_equal_0, y, x),
            paddle.where(
                y_not_equal_0,
                paddle.mod(x, y_safe),
                paddle.zeros(y.shape, y.dtype),
            ),
        )
        return (paddle.where(x < y, y, x), paddle.where(x < y, x, y))

    if in_dynamic_mode():
        while _gcd_cond_fn(x, y):
            x, y = _gcd_body_fn(x, y)

        return x
    else:
        check_variable_and_dtype(x, 'x', ['int32', 'int64'], 'gcd')
        check_variable_and_dtype(y, 'y', ['int32', 'int64'], 'gcd')
        out, _ = paddle.static.nn.while_loop(_gcd_cond_fn, _gcd_body_fn, [x, y])
        return out


def lcm(x, y, name=None):
    """
    Computes the element-wise least common multiple (LCM) of input |x| and |y|.
    Both x and y must have integer types.

    Note:
        lcm(0,0)=0, lcm(0, y)=0

        If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Args:
        x (Tensor): An N-D Tensor, the data type is int32, int64.
        y (Tensor): An N-D Tensor, the data type is int32, int64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.to_tensor(12)
            >>> x2 = paddle.to_tensor(20)
            >>> paddle.lcm(x1, x2)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        60)

            >>> x3 = paddle.arange(6)
            >>> paddle.lcm(x3, x2)
            >>> # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [0, 20, 20, 60, 20, 20])

            >>> x4 = paddle.to_tensor(0)
            >>> paddle.lcm(x4, x2)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        0)

            >>> paddle.lcm(x4, x4)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        0)

            >>> x5 = paddle.to_tensor(-20)
            >>> paddle.lcm(x1, x5)
            >>> # Tensor(shape=[], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            >>> #        60)
    """
    d = paddle.gcd(x, y)
    # paddle.mod will raise an error when any element of y is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    d_equal_0 = paddle.equal(d, 0)
    d_safe = paddle.where(d_equal_0, paddle.ones(d.shape, d.dtype), d)
    out = paddle.where(
        d_equal_0, paddle.zeros(d.shape, d.dtype), paddle.abs(x * y) // d_safe
    )
    return out


def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    r"""
    Computes the n-th forward difference along the given axis.
    The first-order differences is computed by using the following formula:

    .. math::

        out[i] = x[i+1] - x[i]

    Higher-order differences are computed by using paddle.diff() recursively.
    Only n=1 is currently supported.

    Args:
        x (Tensor): The input tensor to compute the forward difference on, the data type is float16, float32, float64, bool, int32, int64.
        n (int, optional): The number of times to recursively compute the difference.
                          Only support n=1. Default:1
        axis (int, optional): The axis to compute the difference along. Default:-1
        prepend (Tensor, optional): The tensor to prepend to input along axis before computing the difference.
                                   It's dimensions must be equivalent to that of x,
                                   and its shapes must match x's shape except on axis.
        append (Tensor, optional): The tensor to append to input along axis before computing the difference,
                                   It's dimensions must be equivalent to that of x,
                                   and its shapes must match x's shape except on axis.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output tensor with same dtype with x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([1, 4, 5, 2])
            >>> out = paddle.diff(x)
            >>> print(out)
            >>> # out:
            >>> # [3, 1, -3]

            >>> y = paddle.to_tensor([7, 9])
            >>> out = paddle.diff(x, append=y)
            >>> print(out)
            >>> # out:
            >>> # [3, 1, -3, 5, 2]

            >>> z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> out = paddle.diff(z, axis=0)
            >>> print(out)
            >>> # out:
            >>> # [[3, 3, 3]]
            >>> out = paddle.diff(z, axis=1)
            >>> print(out)
            >>> # out:
            >>> # [[1, 1], [1, 1]]
    """

    if axis < 0:
        axis = axis + len(x.shape)
    if axis > len(x.shape):
        axis = len(x.shape)
    if axis < 0:
        axis = 0
    dtype = x.dtype
    axes = [axis]
    infer_flags = [1 for i in range(len(axes))]
    if in_dynamic_mode():
        has_pend = False
        input_list = []
        if prepend is not None and append is not None:
            input_list = [prepend, x, append]
            has_pend = True
        elif prepend is not None:
            input_list = [prepend, x]
            has_pend = True
        elif append is not None:
            input_list = [x, append]
            has_pend = True
        if has_pend:
            new_input = _C_ops.concat(input_list, axis)
        else:
            new_input = x

        attrs_1 = ()
        attrs_2 = ()

        dim_len = new_input.shape[axis]

        starts_1 = [0]
        attrs_1 += ('starts', starts_1)
        ends_1 = [dim_len - 1]
        attrs_1 += ('ends', ends_1)
        input_front = _C_ops.slice(
            new_input, axes, starts_1, ends_1, infer_flags, []
        )
        starts_2 = [1]
        attrs_2 += ('starts', starts_2)
        ends_2 = [dim_len]
        attrs_2 += ('ends', ends_2)
        input_back = _C_ops.slice(
            new_input, axes, starts_2, ends_2, infer_flags, []
        )

        if x.dtype == paddle.bool:
            return _C_ops.logical_xor(input_back, input_front)
        else:
            return _C_ops.subtract(input_back, input_front)
    else:
        check_variable_and_dtype(
            x,
            'x',
            ['float16', 'float32', 'float64', 'bool', 'int32', 'int64'],
            'diff',
        )
        check_type(axis, 'axis', (int), 'diff')
        helper = LayerHelper('diff', **locals())
        has_pend = False
        input_list = []
        if prepend is not None and append is not None:
            input_list = [prepend, x, append]
            has_pend = True
        elif prepend is not None:
            input_list = [prepend, x]
            has_pend = True
        elif append is not None:
            input_list = [x, append]
            has_pend = True

        if has_pend:
            new_input = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='concat',
                inputs={'X': input_list},
                outputs={'Out': [new_input]},
                attrs={'axis': axis},
            )
        else:
            new_input = x

        dim_len = new_input.shape[axis]
        attrs_1 = {'axes': axes}
        starts_1 = [0]
        ends_1 = [dim_len - 1]
        attrs_1['starts'] = starts_1
        attrs_1['ends'] = ends_1
        input_front = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='slice',
            inputs={'Input': new_input},
            attrs=attrs_1,
            outputs={'Out': input_front},
        )
        attrs_2 = {'axes': axes}
        starts_2 = [1]
        ends_2 = [dim_len]
        attrs_2['starts'] = starts_2
        attrs_2['ends'] = ends_2
        input_back = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='slice',
            inputs={'Input': new_input},
            attrs=attrs_2,
            outputs={'Out': input_back},
        )

        if dtype == paddle.bool:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='logical_xor',
                inputs={"X": input_back, "Y": input_front},
                outputs={"Out": out},
            )
        else:
            out = paddle.tensor.math.subtract(input_back, input_front)
        return out


def angle(x, name=None):
    r"""
    Element-wise angle of complex numbers. For non-negative real numbers, the angle is 0 while
    for negative real numbers, the angle is :math:`\pi`.

    Equation:
        .. math::

            angle(x)=arctan2(x.imag, x.real)

    Args:
        x (Tensor): An N-D Tensor, the data type is complex64, complex128, or float32, float64 .
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: An N-D Tensor of real data type with the same precision as that of x's data type.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
            >>> y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
            >>> z = x + 1j * y
            >>> print(z)
            >>> # Tensor(shape=[4, 4], dtype=complex64, place=Place(cpu), stop_gradient=True,
            >>> #        [[(-2-2j), (-2-1j), (-2+0j), (-2+1j)],
            >>> #         [(-1-2j), (-1-1j), (-1+0j), (-1+1j)],
            >>> #         [-2j    , -1j    ,  0j    ,  1j    ],
            >>> #         [ (1-2j),  (1-1j),  (1+0j),  (1+1j)]])

            >>> theta = paddle.angle(z)
            >>> print(theta)
            >>> # Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [[-2.35619450, -2.67794514,  3.14159274,  2.67794514],
            >>> #         [-2.03444386, -2.35619450,  3.14159274,  2.35619450],
            >>> #         [-1.57079637, -1.57079637,  0.        ,  1.57079637],
            >>> #         [-1.10714877, -0.78539819,  0.        ,  0.78539819]])
    """

    if in_dynamic_mode():
        return _C_ops.angle(x)
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'float16',
                'float32',
                'float64',
                'complex64',
                'complex128',
                'uint16',
            ],
            'angle',
        )
        op_type = "angle"
        helper = LayerHelper(op_type, **locals())
        inputs = {"X": x}
        out = helper.create_variable_for_type_inference(
            dtype=_complex_to_real_dtype(x.dtype)
        )
        outputs = {"Out": out}
        helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
        return out


def heaviside(x, y, name=None):
    r"""
    Computes the Heaviside step function determined by corresponding element in y for each element in x. The equation is

    .. math::
        heaviside(x, y)=
            \left\{
                \begin{array}{lcl}
                0,& &\text{if} \ x < 0, \\
                y,& &\text{if} \ x = 0, \\
                1,& &\text{if} \ x > 0.
                \end{array}
            \right.

    Note:
        ``paddle.heaviside`` supports broadcasting. If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        x (Tensor): The input tensor of Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        y (Tensor): The tensor that determines a Heaviside step function, it's data type should be float16, float32, float64, int32 or int64.
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. A location into which the result is stored. If x and y have different shapes and are broadcastable, the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([-0.5, 0, 0.5])
            >>> y = paddle.to_tensor([0.1])
            >>> paddle.heaviside(x, y)
            >>> #    [0.        , 0.10000000, 1.        ]
            >>> x = paddle.to_tensor([[-0.5, 0, 0.5], [-0.5, 0.5, 0]])
            >>> y = paddle.to_tensor([0.1, 0.2, 0.3])
            >>> paddle.heaviside(x, y)
            >>> #    [[0.        , 0.20000000, 1.        ],
            >>> #     [0.        , 1.        , 0.30000001]]
    """
    if in_dynamic_mode():
        return _C_ops.heaviside(x, y)
    else:
        op_type = 'elementwise_heaviside'
        return _elementwise_op(LayerHelper(op_type, **locals()))


def frac(x, name=None):
    """
    This API is used to return the fractional portion of each element in input.

    Args:
        x (Tensor): The input tensor, which data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor of frac.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[12.22000003, -1.02999997],
            ...                         [-0.54999995, 0.66000003]])
            >>> output = paddle.frac(input)
            >>> print(output)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [[ 0.22000003, -0.02999997],
            >>> #         [-0.54999995,  0.66000003]])
    """
    if x.dtype not in [
        paddle.int32,
        paddle.int64,
        paddle.float32,
        paddle.float64,
    ]:
        raise TypeError(
            "The data type of input must be one of ['int32', 'int64', 'float32', 'float64'], but got {}".format(
                x.dtype
            )
        )
    if in_dynamic_mode():
        y = _C_ops.trunc(x)
        return _C_ops.subtract(x, y)
    else:
        inputs = {"X": x}
        attrs = {}

        helper = LayerHelper("trunc", **locals())
        check_variable_and_dtype(
            x, "X", ['int32', 'int64', 'float32', 'float64'], 'trunc'
        )
        y = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="trunc", inputs=inputs, attrs=attrs, outputs={"Out": y}
        )
        return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


def sgn(x, name=None):
    """
    For complex tensor, this API returns a new tensor whose elements have the same angles as the corresponding
    elements of input and absolute values of one.
    For other float dtype tensor,
    this API returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero, same as paddle.sign.

    Args:
        x (Tensor): The input tensor, which data type should be float16, float32, float64, complex64, complex128.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A sign Tensor for real input, or normalized Tensor for complex input, shape and data type are same as input.

    Examples:
        .. code-block:: Python

            import paddle

            x = paddle.to_tensor([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]])
            print(paddle.sgn(x))
            #[[0.6+0.8j       0.28-0.96j      0.+0.j      0.4472136+0.8944272j]
            # [0.6+0.8j       1.+0.j          0.+0.j      -1.+0.j]]

    """
    if x.dtype not in [
        paddle.float16,
        paddle.float32,
        paddle.float64,
        paddle.complex64,
        paddle.complex128,
    ]:
        raise TypeError(
            "The data type of input must be one of ['float16', 'float32', 'float64', 'complex64', 'complex128'], but got {}".format(
                x.dtype
            )
        )
    if paddle.is_complex(x):
        expand_x = paddle.as_real(x)
        x_abs = paddle.abs(x)
        x_abs = paddle.unsqueeze(x_abs, axis=-1)
        output = expand_x / x_abs
        zeros = paddle.zeros_like(output)
        output = paddle.where(paddle.isnan(output), zeros, output)

        return paddle.as_complex(output)
    else:
        return paddle.sign(x)


def take(x, index, mode='raise', name=None):
    """
    Returns a new tensor with the elements of input tensor x at the given index.
    The input tensor is treated as if it were viewed as a 1-D tensor.
    The result takes the same shape as the index.

    Args:
        x (Tensor): An N-D Tensor, its data type should be int32, int64, float32, float64.
        index (Tensor): An N-D Tensor, its data type should be int32, int64.
        mode (str, optional): Specifies how out-of-bounds index will behave. the candicates are ``'raise'``, ``'wrap'`` and ``'clip'``.

            - ``'raise'``: raise an error (default);
            - ``'wrap'``: wrap around;
            - ``'clip'``: clip to the range. ``'clip'`` mode means that all indices that are too large are replaced by the index that addresses the last element. Note that this disables indexing with negative numbers.

        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Tensor with the same shape as index, the data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x_int = paddle.arange(0, 12).reshape([3, 4])
            >>> x_float = x_int.astype(paddle.float64)

            >>> idx_pos = paddle.arange(4, 10).reshape([2, 3])  # positive index
            >>> idx_neg = paddle.arange(-2, 4).reshape([2, 3])  # negative index
            >>> idx_err = paddle.arange(-2, 13).reshape([3, 5])  # index out of range

            >>> paddle.take(x_int, idx_pos)
            >>> # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[4, 5, 6],
            >>> #         [7, 8, 9]])

            >>> paddle.take(x_int, idx_neg)
            >>> # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[10, 11, 0 ],
            >>> #         [1 , 2 , 3 ]])

            >>> paddle.take(x_float, idx_pos)
            >>> # Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        [[4., 5., 6.],
            >>> #         [7., 8., 9.]])

            >>> x_int.take(idx_pos)
            >>> # Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
            >>> #        [[4, 5, 6],
            >>> #         [7, 8, 9]])

            >>> paddle.take(x_int, idx_err, mode='wrap')
            >>> # Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
            >>> #        [[10, 11, 0 , 1 , 2 ],
            >>> #         [3 , 4 , 5 , 6 , 7 ],
            >>> #         [8 , 9 , 10, 11, 0 ]])

            >>> paddle.take(x_int, idx_err, mode='clip')
            >>> # Tensor(shape=[3, 5], dtype=int32, place=Place(cpu), stop_gradient=True,
            >>> #        [[0 , 0 , 0 , 1 , 2 ],
            >>> #         [3 , 4 , 5 , 6 , 7 ],
            >>> #         [8 , 9 , 10, 11, 11]])

    """
    if mode not in ['raise', 'wrap', 'clip']:
        raise ValueError(
            "'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {}.".format(
                mode
            )
        )

    if in_dynamic_mode():
        if not isinstance(index, (paddle.Tensor, Variable)):
            raise TypeError(
                "The type of 'index' must be Tensor, but got {}".format(
                    type(index)
                )
            )
        if index.dtype not in [paddle.int32, paddle.int64]:
            raise TypeError(
                "The data type of 'index' must be one of ['int32', 'int64'], but got {}".format(
                    index.dtype
                )
            )

    else:
        check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'take')

    input_1d = x.flatten()
    index_1d = index.flatten()
    max_index = input_1d.shape[-1]

    if mode == 'raise':
        # This processing enables 'take' to handle negative indexes within the correct range.
        index_1d = paddle.where(index_1d < 0, index_1d + max_index, index_1d)
    elif mode == 'wrap':
        # The out of range indices are constrained by taking the remainder.
        index_1d = paddle.where(index_1d < 0, index_1d % max_index, index_1d)
        index_1d = paddle.where(
            index_1d >= max_index, index_1d % max_index, index_1d
        )
    elif mode == 'clip':
        # 'clip' mode disables indexing with negative numbers.
        index_1d = clip(index_1d, 0, max_index - 1)

    out = input_1d.index_select(index_1d).reshape(index.shape)

    return out


def frexp(x, name=None):
    """
    The function used to decompose a floating point number into mantissa and exponent.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    Returns:

        - mantissa (Tensor), A mantissa Tensor. The shape and data type of mantissa tensor and exponential tensor are
            the same as those of input.

        - exponent (Tensor), A exponent Tensor. The shape and data type of mantissa tensor and exponential tensor are
            the same as those of input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1, 2, 3, 4]], dtype="float32")
            >>> print(paddle.tensor.math.frexp(x))
            >>> # (Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,[[0.50000000, 0.50000000, 0.75000000, 0.50000000]]),
            >>> #  Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,[[1., 2., 2., 3.]]))
    """
    if x.dtype not in [paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be one of ['float32', 'float64'], but got {}".format(
                x.dtype
            )
        )
    input_x = paddle.abs(x)
    exponent = paddle.floor(paddle.log2(input_x))
    exponent = paddle.where(
        paddle.isinf(exponent), paddle.full_like(exponent, 0), exponent
    )

    # 0填充
    mantissa = paddle.divide(input_x, 2**exponent)
    # 计算exponent
    exponent = paddle.where(
        (mantissa >= 1),
        paddle.add(exponent, paddle.ones_like(exponent)),
        exponent,
    )
    mantissa = paddle.where(
        (mantissa >= 1),
        paddle.divide(mantissa, 2 ** paddle.ones_like(exponent)),
        mantissa,
    )

    mantissa = paddle.where((x < 0), mantissa * -1, mantissa)
    return mantissa, exponent


def _trapezoid(y, x=None, dx=None, axis=-1, mode='sum'):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        sum_mode (str): use a different summation. The default is `sum`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
    """
    if mode == 'sum':
        sum_mode = paddle.sum
    elif mode == 'cumsum':
        sum_mode = paddle.cumsum

    if not (x is None or dx is None):
        raise ValueError("Not permitted to specify both x and dx input args.")
    if y.dtype not in [paddle.float16, paddle.float32, paddle.float64]:
        raise TypeError(
            "The data type of input must be Tensor, and dtype should be one of ['paddle.float16', 'paddle.float32', 'paddle.float64'], but got {}".format(
                y.dtype
            )
        )

    y_shape = y.shape
    length = y_shape[axis]
    if axis < 0:
        axis += y.dim()
    if x is None:
        if dx is None:
            dx = 1.0
        dx = paddle.to_tensor(dx)
        if dx.dim() > 1:
            raise ValueError(f'Expected dx to be a scalar, got dx={dx}')
    else:
        if x.dtype not in [paddle.float16, paddle.float32, paddle.float64]:
            raise TypeError(
                "The data type of input must be Tensor, and dtype should be one of ['paddle.float16', 'paddle.float32', 'paddle.float64'], but got {}".format(
                    x.dtype
                )
            )
        # Reshape to correct shape
        if x.dim() == 1:
            dx = paddle.diff(x)
            shape = [1] * y.dim()
            shape[axis] = dx.shape[0]
            dx = dx.reshape(shape)
        else:
            dx = paddle.diff(x, axis=axis)
    return 0.5 * sum_mode(
        (
            paddle.gather(y, paddle.arange(1, length), axis=axis)
            + paddle.gather(y, paddle.arange(0, length - 1), axis=axis)
        )
        * dx,
        axis=axis,
    )


def trapezoid(y, x=None, dx=None, axis=-1, name=None):
    """
    Integrate along the given axis using the composite trapezoidal rule. Use the sum method.

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
        If :attr:`y` is a 1D tensor, then the result is a float. If N is greater than 1, then the result is an (N-1)-D tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')

            >>> print(paddle.trapezoid(y))
            >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        10.)

            >>> print(paddle.trapezoid(y, dx=2.))
            >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        20.)

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> print(paddle.trapezoid(y, x))
            >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        10.)


            >>> y = paddle.to_tensor([1, 2, 3], dtype='float64')
            >>> x = paddle.to_tensor([8, 6, 4], dtype='float64')

            >>> print(paddle.trapezoid(y, x))
            >>> # Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        -8.)
            >>> y = paddle.arange(6).reshape((2, 3)).astype('float32')

            >>> print(paddle.trapezoid(y, axis=0))
            >>> # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [1.50000000, 2.50000000, 3.50000000])
            >>> print(paddle.trapezoid(y, axis=1))
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [2., 8.])
    """
    return _trapezoid(y, x, dx, axis, mode='sum')


def cumulative_trapezoid(y, x=None, dx=None, axis=-1, name=None):
    """
    Integrate along the given axis using the composite trapezoidal rule. Use the cumsum method

    Args:
        y (Tensor): Input tensor to integrate. It's data type should be float16, float32, float64.
        x (Tensor, optional): The sample points corresponding to the :attr:`y` values, the same type as :attr:`y`.
            It is known that the size of :attr:`y` is `[d_1, d_2, ... , d_n]` and :math:`axis=k`, then the size of :attr:`x` can only be `[d_k]` or `[d_1, d_2, ... , d_n ]`.
            If :attr:`x` is None, the sample points are assumed to be evenly spaced :attr:`dx` apart. The default is None.
        dx (float, optional): The spacing between sample points when :attr:`x` is None. If neither :attr:`x` nor :attr:`dx` is provided then the default is :math:`dx = 1`.
        axis (int, optional): The axis along which to integrate. The default is -1.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, Definite integral of :attr:`y` is N-D tensor as approximated along a single axis by the trapezoidal rule.
        The result is an N-D tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')

            >>> print(paddle.cumulative_trapezoid(y))
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [4.50000000, 10.       ])

            >>> print(paddle.cumulative_trapezoid(y, dx=2.))
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [9. , 20.])

            >>> y = paddle.to_tensor([4, 5, 6], dtype='float32')
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')

            >>> print(paddle.cumulative_trapezoid(y, x))
            >>> # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [4.50000000, 10.       ])

            >>> y = paddle.to_tensor([1, 2, 3], dtype='float64')
            >>> x = paddle.to_tensor([8, 6, 4], dtype='float64')

            >>> print(paddle.cumulative_trapezoid(y, x))
            >>> # Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            >>> #        [-3., -8.])

            >>> y = paddle.arange(6).reshape((2, 3)).astype('float32')

            >>> print(paddle.cumulative_trapezoid(y, axis=0))
            >>> # Tensor(shape=[1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [[1.50000000, 2.50000000, 3.50000000]])
            >>> print(paddle.cumulative_trapezoid(y, axis=1))
            >>> # Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #        [[0.50000000, 2.        ],
            >>> #         [3.50000000, 8.        ]])
    """
    return _trapezoid(y, x, dx, axis, mode='cumsum')


def vander(x, n=None, increasing=False, name=None):
    """
    Generate a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector. Order of the powers is
    determined by the increasing Boolean parameter. Specifically, when the increment is
    "false", the ith output column is a step-up in the order of the elements of the input
    vector to the N - i - 1 power. Such a matrix with a geometric progression in each row
    is named after Alexandre-Theophile Vandermonde.

    Args:
        x (Tensor): The input tensor, it must be 1-D Tensor, and it's data type should be ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'].
        n (int): Number of columns in the output. If n is not specified, a square array is returned (n = len(x)).
        increasing(bool): Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
    Returns:
        Tensor, A vandermonde matrix with shape (len(x), N). If increasing is False, the first column is :math:`x^{(N-1)}`, the second :math:`x^{(N-2)}` and so forth.
        If increasing is True, the columns are :math:`x^0`, :math:`x^1`, ..., :math:`x^{(N-1)}`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([1., 2., 3.], dtype="float32")
            >>> out = paddle.vander(x)
            >>> print(out.numpy())
            >>> # [[1., 1., 1.],
            >>> #  [4., 2., 1.],
            >>> #  [9., 3., 1.]]
            >>> out1 = paddle.vander(x,2)
            >>> print(out1.numpy())
            >>> # [[1., 1.],
            >>> #  [2., 1.],
            >>> #  [3., 1.]]
            >>> out2 = paddle.vander(x, increasing = True)
            >>> print(out2.numpy())
            >>> # [[1., 1., 1.],
            >>> #  [1., 2., 4.],
            >>> #  [1., 3., 9.]]
            >>> real = paddle.to_tensor([2., 4.])
            >>> imag = paddle.to_tensor([1., 3.])
            >>> complex = paddle.complex(real, imag)
            >>> out3 = paddle.vander(complex)
            >>> print(out3.numpy())
            >>> # [[2.+1.j, 1.+0.j],
            >>> #  [4.+3.j, 1.+0.j]]
    """
    check_variable_and_dtype(
        x,
        'x',
        ['complex64', 'complex128', 'float32', 'float64', 'int32', 'int64'],
        'vander',
    )
    if x.dim() != 1:
        raise ValueError(
            "The input of x is expected to be a 1-D Tensor."
            "But now the dims of Input(X) is %d." % x.dim()
        )

    if n is None:
        n = x.shape[0]

    if n < 0:
        raise ValueError("N must be non-negative.")

    res = paddle.empty([x.shape[0], n], dtype=x.dtype)

    if n > 0:
        res[:, 0] = paddle.to_tensor([1], dtype=x.dtype)
    if n > 1:
        res[:, 1:] = x[:, None]
        res[:, 1:] = paddle.cumprod(res[:, 1:], dim=-1)
    res = res[:, ::-1] if not increasing else res
    return res


def nextafter(x, y, name=None):
    r"""
    Return the next floating-point value after input towards other, elementwise.
    The shapes of input and other must be broadcastable.

    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        y (Tensor): An N-D Tensor, the data type is float32, float64.
        name(str, optional):Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> out = paddle.nextafter(paddle.to_tensor([1.0,2.0]),paddle.to_tensor([2.0,1.0]))
            >>> print(out)
            >>> #Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            >>> #       [1.00000012, 1.99999988])
    """
    if in_dynamic_mode():
        return _C_ops.nextafter(x, y)
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'nextafter')
        check_variable_and_dtype(y, 'y', ['float32', 'float64'], 'nextafter')
        op_type = "nextafter"
        helper = LayerHelper(op_type, **locals())
        inputs = {"x": x, "y": y}
        out = helper.create_variable_for_type_inference(dtype=paddle.float32)
        outputs = {"out": out}
        helper.append_op(type=op_type, inputs=inputs, outputs=outputs)
    return out


def i0(x, name=None):
    r"""
    The function used to calculate modified bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2}

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the modified bessel function of order 0 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i0(x))
            >>> # (Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True, [0.99999994 , 1.26606596 , 2.27958512 , 4.88079262 , 11.30192089]),
    """
    if in_dynamic_mode():
        return _C_ops.i0(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i0")

        helper = LayerHelper("i0", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='i0', inputs={'x': x}, outputs={'out': out})
    return out


def i0e(x, name=None):
    r"""
    The function used to calculate exponentially scaled modified Bessel function of order 0.

    Equation:
        ..  math::

            I_0(x) = \sum^{\infty}_{k=0}\frac{(x^2/4)^k}{(k!)^2} \\
            I_{0e}(x) = e^{-|x|}I_0(x)

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 0 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i0e(x))
            >>> # (Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True, [1., 0.46575961, 0.30850832, 0.24300035, 0.20700192]),
    """
    if in_dynamic_mode():
        return _C_ops.i0e(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i0e")

        helper = LayerHelper("i0e", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(type='i0e', inputs={'x': x}, outputs={'out': out})
    return out


def i1(x, name=None):
    """
    The function is used to calculate modified bessel function of order 1.

    Args:
        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the modified bessel function of order 1 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1(x))
            >>> # (Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True, [0., 0.5651591 , 1.59063685 , 3.95337022 , 9.75946515]),
    """
    if in_dynamic_mode():
        return _C_ops.i1(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i1")

        helper = LayerHelper("i1", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='i1', inputs={'x': x}, outputs={'out': out}, attrs={}
        )
    return out


def i1e(x, name=None):
    """
    The function is used to calculate exponentially scaled modified Bessel function of order 1.

    Args:

        x (Tensor): The input tensor, it's data type should be float32, float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        - out (Tensor), A Tensor. the value of the exponentially scaled modified Bessel function of order 1 at x.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32")
            >>> print(paddle.i1e(x))
            >>> # (Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True, [0., 0.20791042, 0.21526929, 0.24300035, 0.17875084]),
    """
    if in_dynamic_mode():
        return _C_ops.i1e(x)
    else:
        check_variable_and_dtype(x, "x", ["float32", "float64"], "i1e")

        helper = LayerHelper("i1e", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='i1e', inputs={'x': x}, outputs={'out': out}, attrs={}
        )
    return out


def polygamma(x, n, name=None):
    r"""
    Calculates the polygamma of the given input tensor, element-wise.

    The equation is:

    .. math::
        \Phi^n(x) = \frac{d^n}{dx^n} [\ln(\Gamma(x))]

    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        n (int): Order of the derivative. Must be integral.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - out (Tensor), A Tensor. the polygamma of the input Tensor, the shape and data type is the same with input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> data = paddle.to_tensor([2, 3, 25.5], dtype='float32')
            >>> res = paddle.polygamma(data, 1)
            >>> print(res)
            >>> # Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #       [0.64493407,  0.39493407,  0.03999467])
    """
    if not isinstance(n, int):
        raise TypeError(
            "The input of n must be int type, but received: %s " % (type(n))
        )
    if n < 0:
        raise ValueError(
            "The input of n must be greater than or equal to 0. But received n = %s"
            % (n)
        )
    if n == 0:
        return digamma(x)
    else:
        if in_dynamic_mode():
            return _C_ops.polygamma(x, n)
        else:
            check_variable_and_dtype(
                x, "x", ["float32", "float64"], "polygamma"
            )

            helper = LayerHelper("polygamma", **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type='polygamma',
                inputs={'x': x},
                outputs={'out': out},
                attrs={'n': n},
            )
        return out


def ldexp(x, y, name=None):
    """
    Compute the result of multiplying x by 2 to the power of y. The equation is:

    .. math::
        out = x * 2^{y}

    Args:
        x (Tensor): The input Tensor, the data type is float32, float64, int32 or int64.
        y (Tensor):  A Tensor of exponents, typically integers.
        name (str, optional): Name for the operation (optional, default is None).For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): An N-D Tensor. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape, its shape is the same as x and y. And the data type is float32 or float64.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> #example1
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')
            >>> y = paddle.to_tensor([2, 3, 4], dtype='int32')
            >>> res = paddle.ldexp(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [4., 16., 48.])

            >>> #example2
            >>> x = paddle.to_tensor([1, 2, 3], dtype='float32')
            >>> y = paddle.to_tensor([2], dtype='int32')
            >>> res = paddle.ldexp(x, y)
            >>> print(res)
            >>> # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            >>> #        [4., 8., 12.])

    """
    if not isinstance(x, (paddle.Tensor, Variable)):
        raise TypeError(f"x must be tensor type, but got {type(x)}")
    if not isinstance(y, (paddle.Tensor, Variable)):
        raise TypeError(f"y must be tensor type, but got {type(y)}")
    if x.dtype == paddle.float64 or y.dtype == paddle.float64:
        out_dtype = paddle.float64
    else:
        out_dtype = paddle.get_default_dtype()
    x = paddle.cast(x, dtype=out_dtype)
    y = paddle.cast(y, dtype=out_dtype)
    two = paddle.to_tensor(2, dtype=out_dtype)
    return paddle.multiply(x, paddle.pow(two, y))


#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

# TODO: define loss functions of neural network
from paddle import fluid, in_dynamic_mode
from paddle.fluid.framework import in_dygraph_mode

from .. import functional as F
from .layers import Layer

__all__ = []


class BCEWithLogitsLoss(Layer):
    r"""

    Combine the sigmoid layer and the :ref:`api_paddle_nn_BCELoss` layer.

    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.

    Firstly, calculate loss function as follows:

    .. math::
           Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \log(1 + e^{-Logit})

    For stability and to prevent overflow of :math:`e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

        .. math::
           Out = \max(Logit, 0) - Logit * Labels + \log(1 + e^{-\|Logit\|})

    Then, if ``weight`` or ``pos_weight`` is not None, then multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.

    Finally, apply reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target labels ``label`` should be numbers between 0 and 1.

    Args:
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector
            with length equal to the number of classes. The data type is float32, float64.
            Default is ``'None'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
        - logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, `*`], N is batch_size, `*` means number of additional dimensions. The ``logit`` is usually the output of Linear layer. Available dtype is float32, float64.
        - label (Tensor): The target labels tensor. 2-D tensor with the same shape as ``logit``. The target labels which values should be numbers between 0 and 1. Available dtype is float32, float64.
        - output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``logit`` , else the shape of output is scalar.

    Returns:
        A callable object of BCEWithLogitsLoss.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
            >>> bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
            >>> output = bce_logit_loss(logit, label)
            >>> print(output)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        0.45618814)

    """

    def __init__(
        self, weight=None, reduction='mean', pos_weight=None, name=None
    ):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )

        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.name = name

    def forward(self, logit, label):
        out = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            self.weight,
            self.reduction,
            self.pos_weight,
            self.name,
        )
        return out


class CrossEntropyLoss(Layer):
    r"""

    By default, the cross entropy loss function is implemented using softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable computing.

    Calculate the cross entropy loss function without softmax when use_softmax=False.

    By default, calculate the mean of the result, and you can also affect
    the default behavior by using the reduction parameter. Please refer to the part of
    parameters for details.

    Can be used to calculate the softmax cross entropy loss with soft and hard labels.
    Where, the hard labels mean the actual label value, 0, 1, 2, etc.  And the soft labels
    mean the probability of the actual label, 0.6, 0.8, 0.2, etc.

    The calculation includes the following two steps.

    -  **I.softmax cross entropy**

        1. Hard label (each sample can only be assigned into one category)

        1.1. when use_softmax=True

            .. math::
              \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        1.2. when use_softmax=False

            .. math::
              \\loss_j=-\log\left({P}_{label_j}\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).


        2. Soft label (each sample is assigned to multiple categories with a certain probability, and the probability sum is 1).

        2.1. when use_softmax=True

            .. math::
              \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        2.2. when use_softmax=False

            .. math::
              \\loss_j=-\sum_{j=0}^{C}\left({label}_j*\log\left({P}_{label_j}\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).



    -  **II.Weight and reduction processing**

        1. Weight

            If the ``weight`` parameter is ``None`` , go to the next step directly.

            If the ``weight`` parameter is not ``None`` , the cross entropy of each sample is weighted by weight
            according to soft_label = False or True as follows.

            1.1. Hard labels (soft_label = False)

            .. math::
                \\loss_j=loss_j*weight[label_j]


            1.2. Soft labels (soft_label = True)

             .. math::
                \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

        2. reduction

            2.1 if the ``reduction`` parameter is ``none``

            Return the previous result directly

            2.2 if the ``reduction`` parameter is ``sum``

            Return the sum of the previous results

            .. math::
               \\loss=\sum_{j}loss_j

            2.3 if the ``reduction`` parameter is ``mean`` , it will be processed according to
            the ``weight`` parameter as follows.

            2.3.1. If the  ``weight``  parameter is ``None``

            Return the average value of the previous results

             .. math::
                \\loss=\sum_{j}loss_j/N

            where, N is the number of samples and C is the number of categories.

            2.3.2. If the 'weight' parameter is not 'None', the weighted average value of the previous result will be returned

            1. Hard labels (soft_label = False)

             .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

            2. Soft labels (soft_label = True)

             .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)


    Parameters:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size C and the data type is float32, float64.
            Default is ``'None'`` .
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the loss. A negative value means that no label
            value needs to be ignored. Only valid when soft_label = False.
            Default is ``-100`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        soft_label (bool, optional): Indicate whether label is soft.
            If soft_label=False, the label is hard.  If soft_label=True, the label is soft.
            Default is ``False``.
        axis (int, optional): The index of dimension to perform softmax calculations.
            It should be in range :math:`[-1, rank - 1]`, where :math:`rank` is the number
            of dimensions of input :attr:`input`.
            Default is ``-1`` .
        use_softmax (bool, optional): Indicate whether compute softmax before cross_entropy.
            Default is ``True``.
        name (str, optional): The name of the operator. Default is ``None`` .
            For more information, please refer to :ref:`api_guide_Name` .


    Shape:
        - **input** (Tensor), the data type is float32, float64. Shape is :math:`[N_1, N_2, ..., N_k, C]`, where C is number of classes, ``k >= 1`` .

            Note:

                1. when use_softmax=True, it expects unscaled logits. This operator should not be used with the
                output of softmax operator, which will produce incorrect results.

                2. when use_softmax=False, it expects the output of softmax operator.

        - **label** (Tensor)

            1. If soft_label=False, the shape is
            :math:`[N_1, N_2, ..., N_k]` or :math:`[N_1, N_2, ..., N_k, 1]`, k >= 1.
            the data type is int32, int64, float32, float64, where each value is [0, C-1].

            2. If soft_label=True, the shape and data type should be same with ``input`` ,
            and the sum of the labels for each sample should be 1.

        - **output** (Tensor), Return the softmax cross_entropy loss of ``input`` and ``label``.
          The data type is the same as input.
          If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the dimension of return value is ``1``.
          If :attr:`reduction` is ``'none'``:

            1. If soft_label = False, the dimension of return value is the same with ``label`` .

            2. if soft_label = True, the dimension of return value is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:

        .. code-block:: python

            >>> # hard labels
            >>> import paddle
            >>> paddle.seed(99999)
            >>> N=100
            >>> C=200
            >>> reduction='mean'
            >>> input =  paddle.rand([N, C], dtype='float64')
            >>> label =  paddle.randint(0, C, shape=[N], dtype='int64')
            >>> weight = paddle.rand([C], dtype='float64')

            >>> cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            ...     weight=weight, reduction=reduction)
            >>> dy_ret = cross_entropy_loss(
            ...                             input,
            ...                             label)
            >>> print(dy_ret)
            >>> # Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        5.34043430)

        .. code-block:: python

            >>> # soft labels
            >>> import paddle
            >>> paddle.seed(99999)
            >>> axis = -1
            >>> ignore_index = -100
            >>> N = 4
            >>> C = 3
            >>> shape = [N, C]
            >>> reduction='mean'
            >>> weight = None
            >>> logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> labels /= paddle.sum(labels, axis=axis, keepdim=True)
            >>> paddle_loss_mean = paddle.nn.functional.cross_entropy(
            ...                                                         logits,
            ...                                                         labels,
            ...                                                         soft_label=True,
            ...                                                         axis=axis,
            ...                                                         weight=weight,
            ...                                                         reduction=reduction)
            >>> print(paddle_loss_mean)
            >>> # Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        1.11043464)

    """

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction='mean',
        soft_label=False,
        axis=-1,
        use_softmax=True,
        name=None,
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.soft_label = soft_label
        self.axis = axis
        self.use_softmax = use_softmax
        self.name = name

    def forward(self, input, label):
        ret = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            use_softmax=self.use_softmax,
            name=self.name,
        )

        return ret


class HSigmoidLoss(Layer):
    """
    Hierarchical Sigmoid Layer.

    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>_`. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        feature_size (int): The number of features.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight_attr (ParamAttr, optional): The parameter attribute for the learnable weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default is None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, `path_table` and
            `path_code` should be passed to its forward method, otherwise `path_table` and `path_code`
            should not be passed to its forward method. Default is False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True,
            the gradient of weight and input will be sparse. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input (Tensor): The input tensor. The shapes is [N, D], where N is batch size and D is feature size. It's data type should be float32, float64.
        label (Tensor): It's shapes is [N, 1]. It's data type should be int64.
        output (Tensor): The HSigmoid Loss of ``input`` and ``label``. Shape is [N, 1]

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.set_device('cpu')

            >>> input = paddle.uniform([4, 3])
            >>> # [[0.56194401  -0.22450298  -0.10741806] # random
            >>> #  [0.36136317  0.23556745  0.88748658] # random
            >>> #  [0.18151939  0.80947340  -0.31078976] # random
            >>> #  [0.68886101  -0.14239830  -0.41297770]] # random
            >>> label = paddle.to_tensor([0, 1, 4, 5])
            >>> m = paddle.nn.HSigmoidLoss(3, 5)
            >>> out = m(input, label)
            >>> # [[2.42524505]
            >>> #  [1.74917245]
            >>> #  [3.14571381]
            >>> #  [2.34564662]]
    """

    def __init__(
        self,
        feature_size,
        num_classes,
        weight_attr=None,
        bias_attr=None,
        is_custom=False,
        is_sparse=False,
        name=None,
    ):
        super().__init__()
        if (num_classes < 2) and (not is_custom):
            raise ValueError(
                "num_classes must not be less than 2 with default tree"
            )

        if (not is_custom) and (is_sparse):
            print("Sparse mode should not be used without custom tree")
            is_sparse = False

        self._feature_size = feature_size
        self._num_classes = num_classes
        self._is_custom = is_custom
        self._is_sparse = is_sparse

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        self._name = name
        self._dtype = paddle.get_default_dtype()

        remote_prefetch = is_sparse
        print(
            "With sparse mode, if your models has only"
            " small parameter prefetch may cause speed down"
        )

        C = self._num_classes if is_custom else self._num_classes - 1
        self.weight = self.create_parameter(
            [C, self._feature_size],
            attr=self._weight_attr,
            is_bias=False,
            dtype=self._dtype,
        )
        self.bias = self.create_parameter(
            [C, 1], attr=self._bias_attr, is_bias=True, dtype=self._dtype
        )

    def forward(self, input, label, path_table=None, path_code=None):
        out = F.hsigmoid_loss(
            input,
            label,
            self._num_classes,
            self.weight,
            self.bias,
            path_table=path_table,
            path_code=path_code,
            is_sparse=self._is_sparse,
            name=self._name,
        )
        return out


class MSELoss(Layer):
    r"""
    **Mean Square Error Loss**
    Computes the mean square error (squared L2 norm) of given input and label.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    where `input` and `label` are `float32` tensors of same shape.

    Parameters:
        reduction (str, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.

    Shape:
        input (Tensor): Input tensor, the data type is float32 or float64
        label (Tensor): Label tensor, the data type is float32 or float64
        output (Tensor): output tensor storing the MSE loss of input and label, the data type is same as input.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> mse_loss = paddle.nn.loss.MSELoss()
            >>> input = paddle.to_tensor([1.5])
            >>> label = paddle.to_tensor([1.7])
            >>> output = mse_loss(input, label)
            >>> print(output)
            >>> # 0.04000002

    """

    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MSELoss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(reduction)
            )
        self.reduction = reduction

    def forward(self, input, label):
        if not in_dynamic_mode():
            fluid.data_feeder.check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'MSELoss'
            )
            fluid.data_feeder.check_variable_and_dtype(
                label, 'label', ['float32', 'float64'], 'MSELoss'
            )

        if in_dygraph_mode():
            square_out = paddle._C_ops.square(paddle.subtract(input, label))
        else:
            square_out = paddle.square(paddle.subtract(input, label))
        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            square_out = paddle.sum(square_out)
            return square_out

        return paddle.mean(square_out)


class L1Loss(Layer):
    r"""

    Construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label\rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label\rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label\rvert)


    Parameters:
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): The input tensor. The shapes is ``[N, *]``, where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        - label (Tensor): label. The shapes is ``[N, *]``, same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        - output (Tensor): The L1 Loss of ``input`` and ``label``.
          If `reduction` is ``'none'``, the shape of output loss is ``[N, *]``, the same as ``input`` .
          If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
            >>> label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

            >>> l1_loss = paddle.nn.L1Loss()
            >>> output = l1_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        0.34999999)

            >>> l1_loss = paddle.nn.L1Loss(reduction='sum')
            >>> output = l1_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        1.39999998)

            >>> l1_loss = paddle.nn.L1Loss(reduction='none')
            >>> output = l1_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[0.20000005, 0.19999999],
            >>> #         [0.20000000, 0.79999995]])

    """

    def __init__(self, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        super().__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return paddle.nn.functional.l1_loss(
            input, label, self.reduction, name=self.name
        )


class BCELoss(Layer):
    """

    This interface is used to construct a callable object of the ``BCELoss`` class.
    The BCELoss layer measures the binary_cross_entropy loss between input predictions ``input``
    and target labels ``label`` . The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the interface will return the original loss `Out`.

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions ``input`` always be the output of sigmoid, and the target labels ``label``
    should be numbers between 0 and 1.

    Parameters:
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, has to be a Tensor of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): 2-D tensor with shape: ``[N, *]``, N is batch_size, `*` means number of additional dimensions. The input ``input`` should always be the output of sigmod. Available dtype is float16, float32, float64.
        - label (Tensor): 2-D tensor with the same shape as ``input``. The target labels which values should be numbers between 0 and 1. Available dtype is float16, float32, float64.
        - output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``input`` , else the shape of output is scalar.

    Returns:
        A callable object of BCELoss.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([0.5, 0.6, 0.7])
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0])
            >>> bce_loss = paddle.nn.BCELoss()
            >>> output = bce_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        0.65537101)

    """

    def __init__(self, weight=None, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )

        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        out = paddle.nn.functional.binary_cross_entropy(
            input, label, self.weight, self.reduction, self.name
        )
        return out


class NLLLoss(Layer):
    r"""

    This class accepts input and target label and returns negative log likelihood
    cross error. It is useful to train a classification problem with C classes.

    The input for the loss is expected to contain log-probabilities of
    each classes. It has to be a Tensor of size either (batch_size, C) or
    (batch_size, C, d1, d2, ..., dK) with K >= 1 for the K-dimensional case.
    The label for the loss should be a class index in the range [0, C-1]
    where C is the number of classes. If ignore_index is specified, the
    specified target value does not contribute to the input gradient.

    If the optional argument `weight` is provided, it should be a 1D Tensor
    assigning weight to each of the classed. This is particularly useful
    when you have an unbalanced training set.

    The loss is calculated as follows.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::

        \ell(x, y) =
        \left\{
            \begin{array}{lcl}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if  reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if  reduction} = \text{'sum'.}
            \end{array}
        \right.

    Parameters:
        weight (Tensor, optional): Weight tensor, a manual rescaling weight given
            to each class. If given, it has to be a 1D Tensor whose size is `[C, ]`. Otherwise,
            it treated as if having all ones. the data type is
            float32, float64, Default is ``'None'``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
        reduction (str, optional): Indicate how to average the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``. Default is ``'mean'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be apllied.
            Default is ``'mean'``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default is ``'None'``.

    Shape:
        - input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
            But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
            The data type is float32, float64.
        - label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
            The data type is int64.
        - output (Tensor): the `negative log likelihood loss` between input `x` and `label`.
            If `reduction` is `'none'`, the shape is `[N, *]`.
            If `reduction` is `'sum'` or `'mean'`, the shape is `[]`.

    Examples:
        .. code-block:: python

            ...     import paddle

            ...     nll_loss = paddle.nn.loss.NLLLoss()
            ...     log_softmax = paddle.nn.LogSoftmax(axis=1)

            ...     input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
            ...                                 [0.53331435, 0.07999352, 0.8549948 ],
            ...                                 [0.25879037, 0.39530203, 0.698465  ],
            ...                                 [0.73427284, 0.63575995, 0.18827209],
            ...                                 [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
            ...     log_out = log_softmax(input)
            ...     label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
            ...     result = nll_loss(log_out, label)
            ...     print(result) # Tensor(shape=[], dtype=float32, place=CPUPlace, stop_gradient=True, 1.07202101)

    """

    def __init__(
        self, weight=None, ignore_index=-100, reduction='mean', name=None
    ):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
                "'none', but received %s, which is not allowed." % reduction
            )
        super().__init__()
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._name = name

    def forward(self, input, label):
        return F.nll_loss(
            input,
            label,
            weight=self._weight,
            ignore_index=self._ignore_index,
            reduction=self._reduction,
            name=self._name,
        )


class PoissonNLLLoss(Layer):
    r"""Generate a callable object of 'PoissonNLLLoss' to calculate the
    Poisson negative log likelihood loss between Input(input) and
    Input(label). Notes that Input(input) is the expectation of underlying
    Poisson distribution and Input(label) is the random samples from the
    Poisson distribution


    Poisson negative log likelihood loss is calculated as follows:

    .. math::
        \text{loss}(\text{input}, \text{label}) = \text{input} - \text{label} * \log(\text{label}) + \log(\text{label!})

    The last term can be approximated with Stirling formula. This approximation term is used when :attr:`full` is ``True``.
    The approximation is added when label values are more than 1 and omitted when the labels are less than or equal to 1.

    Parameters:
         log_input (bool, optional):
            Whether to the treat input tensor as log input.
            If ``True`` the loss is computed as, :math:`\exp(\text{input}) - \text{label} * \text{input}` .
            If ``False`` then loss is :math:`\text{input} - \text{label} * \log(\text{input}+\text{epsilon})` .
            Default: ``True``.
         full (bool, optional):
            Whether to compute full loss.
            If ``True``, the Stirling approximation term is added.
            If ``False``, the Stirling approximation is dropped.
            Default: ``False``.
         epsilon (float, optional):
            A small value to avoid evaluation of :math:`\log(0)` when ``log_input`` = ``False``. ``epsilon > 0``.
            Default: 1e-8.
         reduction (str, optional):
            Indicate how to reduce the loss, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be apllied.
            Default is ``'mean'``.
         name (str, optional):
            Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): The shape of input tensor should be `(N, *)` or `(*)` where `(*)` denotes any number of extra dimensions.
        - label (Tensor): The shape of input tensor should be `(N, *)` or `(*)`, same shape as the input tensor.
        - output (Tensor): scalar if :attr:`reduction` is ``'mean'`` (default) or ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as the input

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> poisson_nll_loss = paddle.nn.loss.PoissonNLLLoss()
            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> loss = poisson_nll_loss(input, label)

    """

    def __init__(
        self,
        log_input=True,
        full=False,
        epsilon=1e-8,
        reduction="mean",
        name=None,
    ):
        if epsilon <= 0:
            raise ValueError(
                "The value of `epsilon` in PoissonNLLLoss should be positve, but received %f, which is not allowed"
                % epsilon
            )
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in PoissonNLLLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        super().__init__()
        self._log_input = log_input
        self._full = full
        self._epsilon = epsilon
        self._reduction = reduction
        self._name = name

    def forward(self, input, label):
        return F.poisson_nll_loss(
            input,
            label,
            log_input=self._log_input,
            full=self._full,
            epsilon=self._epsilon,
            reduction=self._reduction,
            name=self._name,
        )


class KLDivLoss(Layer):
    r"""

    Generate a callable object of 'KLDivLoss' to calculate the
    Kullback-Leibler divergence loss between Input(X) and
    Input(Target). Notes that Input(X) is the log-probability
    and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    $$l(x, y) = y * (\log(y) - x)$$

    Here :math:`x` is input and :math:`y` is label.

    If `reduction` is ``'none'``, the output loss is the same shape as the input, and the loss at each point is calculated separately. There is no reduction to the result.

    If `reduction` is ``'mean'``, the output loss is the shape of [], and the output is the average of all losses.

    If `reduction` is ``'sum'``, the output loss is the shape of [], and the output is the sum of all losses.

    If `reduction` is ``'batchmean'``, the output loss is the shape of [N], N is the batch size, and the output is the sum of all losses divided by the batch size.

    Parameters:
        reduction (str, optional): Indicate how to average the loss,
            the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be apllied.
            Default is ``'mean'``.

    Shape:

        input (Tensor): ``(N, *)``, where ``*`` means, any number of additional dimensions.

        label (Tensor): ``(N, *)``, same shape as input.

        output (Tensor): tensor with shape: [] by default.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> shape = (5, 20)
            >>> x = paddle.uniform(shape, min=-10, max=10).astype('float32')
            >>> target = paddle.uniform(shape, min=-10, max=10).astype('float32')

            >>> # 'batchmean' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> # shape=[]

            >>> # 'mean' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='mean')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> # shape=[]

            >>> # 'sum' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='sum')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> # shape=[]

            >>> # 'none' reduction, loss shape is same with X shape
            >>> kldiv_criterion = nn.KLDivLoss(reduction='none')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> # shape=[5, 20]

    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, label):
        out = F.kl_div(input, label, self.reduction)
        return out


class MarginRankingLoss(Layer):
    r"""

    This interface is used to construct a callable object of the ``MarginRankingLoss`` class.
    The MarginRankingLoss layer calculates the margin rank loss between the input, other and label
    , use the math function as follows.

    .. math::
        margin\_rank\_loss = max(0, -label * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        other: N-D Tensor, `other` have the same shape and dtype as `input`.

        label: N-D Tensor, label have the same shape and dtype as `input`.

        output: If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the out shape is :math:`[]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Returns:
        A callable object of MarginRankingLoss.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
            >>> other = paddle.to_tensor([[2, 1], [2, 4]], dtype="float32")
            >>> label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float32")
            >>> margin_rank_loss = paddle.nn.MarginRankingLoss()
            >>> loss = margin_rank_loss(input, other, label)

            >>> print(loss)
            >>> # 0.75
    """

    def __init__(self, margin=0.0, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input, other, label):
        out = paddle.nn.functional.margin_ranking_loss(
            input, other, label, self.margin, self.reduction, self.name
        )
        return out


class CTCLoss(Layer):
    r"""

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is interated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default is 0.
        reduction (string, optional): Indicate how to average the loss, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.

    Shape:
        - log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        - labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        - input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        - label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        - norm_by_times (bool, optional): Whether to normalize the gradients by the number of time-step, which is also the sequence's length. There is no need to normalize the gradients if reduction mode is 'mean'. Default: False.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            >>> # declarative mode
            >>> import paddle

            >>> # length of the longest logit sequence
            >>> max_seq_length = 4
            >>> #length of the longest label sequence
            >>> max_label_length = 3
            >>> # number of logit sequences
            >>> batch_size = 2
            >>> # class num
            >>> class_num = 3

            >>> log_probs = paddle.to_tensor([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
            ...                         [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

            ...                         [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
            ...                         [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

            ...                         [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
            ...                         [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

            ...                         [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
            ...                         [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

            ...                         [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
            ...                         [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]], dtype="float32")
            >>> labels = paddle.to_tensor([[1, 2, 2],
            ...                 [1, 2, 2]], dtype="int32")
            >>> input_lengths = paddle.to_tensor([5, 5], dtype="int64")
            >>> label_lengths = paddle.to_tensor([3, 3], dtype="int64")

            >>> loss = paddle.nn.CTCLoss(blank=0, reduction='none')(log_probs, labels,
            ...     input_lengths,
            ...     label_lengths)
            >>> print(loss)
            >>> # Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [3.91798496, 2.90765190])

            >>> loss = paddle.nn.CTCLoss(blank=0, reduction='mean')(log_probs, labels,
            ...     input_lengths,
            ...     label_lengths)
            >>> print(loss)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        1.13760614)
    """

    def __init__(self, blank=0, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(
        self,
        log_probs,
        labels,
        input_lengths,
        label_lengths,
        norm_by_times=False,
    ):
        return paddle.nn.functional.ctc_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            self.blank,
            self.reduction,
            norm_by_times=norm_by_times,
        )


class RNNTLoss(Layer):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        fastemit_lambda (float, optional): Regularization parameter for FastEmit (https://arxiv.org/pdf/2010.11148.pdf)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'

    Shape:
        input: logprob Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        label: 2 dimensional (batch, labelLength) Tensor containing all the targets of the batch with zero padded
        input_lengths: Tensor of size (batch) containing size of each output sequence from the network
        label_lengths: Tensor of (batch) containing label length of each example

    Returns:
     Tensor, The RNN-T loss between ``logprobs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``logprobs``.

    Examples:
        .. code-block:: python

            >>> # declarative mode
            >>> import numpy as np
            >>> import paddle
            >>> from paddle.nn import RNNTLoss

            >>> fn = RNNTLoss(reduction='sum', fastemit_lambda=0.0)

            >>> acts = np.array([[[[0.1, 0.6, 0.1, 0.1, 0.1],
            ...                 [0.1, 0.1, 0.6, 0.1, 0.1],
            ...                 [0.1, 0.1, 0.2, 0.8, 0.1]],
            ...                 [[0.1, 0.6, 0.1, 0.1, 0.1],
            ...                 [0.1, 0.1, 0.2, 0.1, 0.1],
            ...                 [0.7, 0.1, 0.2, 0.1, 0.1]]]])
            >>> labels = [[1, 2]]

            >>> acts = paddle.to_tensor(acts, stop_gradient=False)

            >>> lengths = [acts.shape[1]] * acts.shape[0]
            >>> label_lengths = [len(l) for l in labels]
            >>> labels = paddle.to_tensor(labels, paddle.int32)
            >>> lengths = paddle.to_tensor(lengths, paddle.int32)
            >>> label_lengths = paddle.to_tensor(label_lengths, paddle.int32)

            >>> costs = fn(acts, labels, lengths, label_lengths)
            >>> print(costs)
            >>> # Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=False,
            >>> #        4.49566677)
    """

    def __init__(
        self, blank=0, fastemit_lambda=0.001, reduction='mean', name=None
    ):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.name = name

    def forward(self, input, label, input_lengths, label_lengths):
        return paddle.nn.functional.rnnt_loss(
            input,
            label,
            input_lengths,
            label_lengths,
            blank=self.blank,
            fastemit_lambda=self.fastemit_lambda,
            reduction=self.reduction,
            name=self.name,
        )


class SmoothL1Loss(Layer):
    r"""
    This operator calculates smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

        loss(x, y) = \frac{1}{n}\sum_{i}z_i

    where :math:`z_i` is given by:

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
                0.5(x_i - y_i)^2 & & {if |x_i - y_i| < \delta} \\
                \delta * |x_i - y_i| - 0.5 * \delta^2 & & {otherwise}
            \end{array} \right.

    Parameters:
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        delta (float, optional): Specifies the hyperparameter :math:`\delta` to be used.
            The value determines how large the errors need to be to use L1. Errors
            smaller than delta are minimized with L2. Parameter is ignored for
            negative/zero values. Default value is :math:`1.0`.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Call Parameters:

        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C),
        where C is number of classes, and if shape is more than 2D,
        this is (N, C, D1, D2,..., Dk), k >= 1.

        label (Tensor): Label tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

    Returns:
        Tensor, The tensor storing the smooth_l1_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> input = paddle.rand([3, 3]).astype("float32")
            >>> label = paddle.rand([3, 3]).astype("float32")
            >>> loss = paddle.nn.SmoothL1Loss()
            >>> output = loss(input, label)
            >>> print(output)
            >>> # 0.049606
    """

    def __init__(self, reduction='mean', delta=1.0, name=None):
        super().__init__()
        self.reduction = reduction
        self.delta = delta
        self.name = name

    def forward(self, input, label):
        return F.smooth_l1_loss(
            input,
            label,
            reduction=self.reduction,
            delta=self.delta,
            name=self.name,
        )


class MultiLabelSoftMarginLoss(Layer):
    r"""Creates a criterion that optimizes a multi-class multi-classification
        hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
        and output :math:`y` (which is a 2D `Tensor` of target class indices).
        For each sample in the mini-batch:

        .. math::
            \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

        where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
        :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
        :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
        and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.
        :math:`y` and :math:`x` must have the same size.

        Parameters:
            weight (Tensor,optional): a manual rescaling weight given to each class.
                    If given, has to be a Tensor of size C and the data type is float32, float64.
                    Default is ``'None'`` .
            reduction (str, optional): Indicate how to average the loss by batch_size,
                    the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
                    If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                    If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                    If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                    Default: ``'mean'``
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.

        Call parameters:
            input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64. The shape of label is the same as the shape of input.

        Shape:
            input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means number of classes, available dtype is float32, float64. The sum operationoperates over all the elements.
            label: N-D Tensor, same shape as the input.
            output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

        Returns:
            A callable object of MultiLabelSoftMarginLoss.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.nn as nn

                >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
                >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

                >>> multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='none')
                >>> loss = multi_label_soft_margin_loss(input, label)
                >>> print(loss)
                >>> # Tensor([3.49625897, 0.71111226, 0.43989015])

                >>> multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
                >>> loss = multi_label_soft_margin_loss(input, label)
                >>> print(loss)
                >>> # Tensor(1.54908717)
        """

    def __init__(self, weight=None, reduction="mean", name=None):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MultiLabelSoftMarginloss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(reduction)
            )
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return F.multi_label_soft_margin_loss(
            input,
            label,
            weight=self.weight,
            reduction=self.reduction,
            name=self.name,
        )


class HingeEmbeddingLoss(Layer):
    r"""
    Create a callable object of `HingeEmbeddingLoss` to calculates hinge_embedding_loss. Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`(containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance as :math:`x`,
    and is typically used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Parameters:

        margin (float, optional): Specifies the hyperparameter margin to be used.
            The value determines how large the input need to be to calculate in
            hinge_embedding_loss. When label is -1, Input smaller than margin are minimized with hinge_embedding_loss.
            Default = 1.0
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Call Parameters:

        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.

        label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64. The shape of label is the same as the shape of input.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64. The sum operationoperates over all the elements.

        label: N-D Tensor, same shape as the input.

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:

        Tensor, The tensor variable storing the hinge_embedding_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> # label elements in {1., -1.}
            >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

            >>> hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='none')
            >>> loss = hinge_embedding_loss(input, label)
            >>> print(loss)
            >>> # Tensor([[0., -2., 0.],
            >>> #         [0., -1., 2.],
            >>> #         [1., 1., 1.]])

            >>> hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
            >>> loss = hinge_embedding_loss(input, label)
            >>> print(loss)
            >>> # Tensor(0.22222222)
    """

    def __init__(self, margin=1.0, reduction="mean", name=None):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return F.hinge_embedding_loss(
            input,
            label,
            reduction=self.reduction,
            margin=self.margin,
            name=self.name,
        )


class CosineEmbeddingLoss(Layer):
    r"""
    This interface is used to construct a callable object of the ``CosineEmbeddingLoss`` class.
    The CosineEmbeddingLoss layer measures the cosine_embedding loss between input predictions ``input1``, ``input2``
    and target labels ``label`` with values 1 or 0. This is used for measuring whether two inputs are similar or
    dissimilar and is typically used for learning nonlinear embeddings or semi-supervised learning.
    The cosine embedding loss can be described as:

    If label = 1, then the loss value can be calculated as follow:

    .. math::
        Out = 1 - cos(input1, input2)

    If label = -1, then the loss value can be calculated as follow:

    .. math::
        Out = max(0, cos(input1, input2)) - margin

    The operator cos can be described as follow:
     .. math::
        cos(x1, x2) = \frac{x1 \cdot{} x2}{\Vert x1 \Vert_2 * \Vert x2 \Vert_2}

    Parameters:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input1 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        input2 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        label (Tensor): tensor with shape: [N] or [1], 'N' means the length of input array. The target labels values should be -1 or 1.
                         Available dtypes are int32, int64, float32, float64.
        output (Tensor): Tensor, the cosine embedding Loss of Tensor ``input1`` ``input2`` and ``label``.
                         If `reduction` is ``'none'``, the shape of output loss is [N], the same as ``input`` .
                         If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
            >>> input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
            >>> label = paddle.to_tensor([1, -1], 'int64')

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output) # 0.21155193

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='sum')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output) # 0.42310387

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output) # [0.42310387, 0.        ]

    """

    def __init__(self, margin=0, reduction='mean', name=None):
        if margin > 1 or margin < -1:
            raise ValueError(
                "The value of 'margin' should be in the interval of [-1, 1], but received %f, which is not allowed."
                % margin
            )
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' should be 'sum', 'mean' or "
                "'none', but received %s, which is not allowed." % reduction
            )
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input1, input2, label):
        return F.cosine_embedding_loss(
            input1,
            input2,
            label,
            margin=self.margin,
            reduction=self.reduction,
            name=self.name,
        )


class TripletMarginWithDistanceLoss(Layer):
    r"""
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}

    where the default `distance_function`

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_2

    or user can define their own distance function. `margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to be 0. If `swap` is true, it will compare distance of (input, negative) with
    distance of (negative, positive) and change it to the smaller one. For more details see http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf.

    Parameters:
        distance_function (Callable, Optional): Quantifies the distance between two tensors. if not specified, 2 norm functions will be used.

        margin (float, Optional):Default: :math:`1`.A nonnegative margin representing the minimum difference
                between the positive and negative distances required for the loss to be 0. Larger
                margins penalize cases where the negative examples are not distant enough from the
                anchors, relative to the positives.

        swap (bool, Optional):The distance swap changes the negative distance to the swap distance (distance between positive samples
                and negative samples) if swap distance smaller than negative distance. Default: ``False``.

        reduction (str, Optional):Indicate how to average the loss by batch_size.
                the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
        input (Tensor):Input tensor, the data type is float32 or float64.
    the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor):Positive tensor, the data type is float32 or float64.
    The shape of label is the same as the shape of input.

        negative (Tensor):Negative tensor, the data type is float32 or float64.
    The shape of label is the same as the shape of input.

        output(Tensor): The tensor variable storing the triplet_margin_with_distance_loss of input and positive and negative.

    Return：
        A callable object of TripletMarginWithDistanceLoss

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn import TripletMarginWithDistanceLoss

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> triplet_margin_with_distance_loss = TripletMarginWithDistanceLoss(reduction='none')
            >>> loss = triplet_margin_with_distance_loss(input, positive, negative,)
            >>> print(loss)
            >>> # Tensor([0.        , 0.57496738, 0.        ])

            >>> triplet_margin_with_distance_loss = TripletMarginWithDistanceLoss(reduction='mean')
            >>> loss = triplet_margin_with_distance_loss(input, positive, negative,)
            >>> print(loss)
            >>> # Tensor(0.19165580)

    """

    def __init__(
        self,
        distance_function=None,
        margin=1.0,
        swap=False,
        reduction: str = 'mean',
        name=None,
    ):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in TripletMarginWithDistanceLoss "
                "should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.distance_function = distance_function
        self.name = name

    def forward(self, input, positive, negative):
        return F.triplet_margin_with_distance_loss(
            input,
            positive,
            negative,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
            name=self.name,
        )


class TripletMarginLoss(Layer):
    r"""
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, *)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Parameters:
        margin (float, Optional):Default: :math:`1`.

        p (int, Optional):The norm degree for pairwise distance. Default: :math:`2`.

        epsilon (float, Optional):Add small value to avoid division by zero,
            default value is 1e-6.

        swap (bool, Optional):The distance swap change the negative distance to the distance between
            positive sample and negative sample. For more details, see `Learning shallow convolutional feature descriptors with triplet losses`.
            Default: ``False``.

        reduction (str, Optional):Indicate how to average the loss by batch_size.
                the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``

        name (str,Optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.

    Call Parameters:
        input (Tensor):Input tensor, the data type is float32 or float64.
        the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor):Positive tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

        negative (Tensor):Negative tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

    Returns:
        Tensor. The tensor variable storing the triplet_margin_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> triplet_margin_loss = paddle.nn.TripletMarginLoss(reduction='none')
            >>> loss = triplet_margin_loss(input, positive, negative)
            >>> print(loss)
            >>> # Tensor([0.        , 0.57496738, 0.        ])

            >>> triplet_margin_loss = paddle.nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean', )
            >>> loss = triplet_margin_loss(input, positive, negative,)
            >>> print(loss)
            >>> # Tensor(0.19165580)

    """

    def __init__(
        self,
        margin=1.0,
        p=2.0,
        epsilon=1e-6,
        swap=False,
        reduction='mean',
        name=None,
    ):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in TripletMarginLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )
        self.margin = margin
        self.p = p
        self.epsilon = epsilon
        self.swap = swap
        self.reduction = reduction
        self.name = name

    def forward(self, input, positive, negative):
        return F.triplet_margin_loss(
            input,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            epsilon=self.epsilon,
            swap=self.swap,
            reduction=self.reduction,
            name=self.name,
        )


class MultiMarginLoss(Layer):
    r"""Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between
    input :math:`input` and label :math:`label`:

    For i-th mini-batch sample, the loss in terms of the 1D input :math:`input_i` and scalar
    output :math:`label_i` is:

    .. math::
        \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, \text{margin} - input_i[label_i] + input_i[j])^p}{\text{C}}

    where :math:`0 \leq j \leq \text{C}-1`, :math:`0 \leq i \leq \text{N}-1` and :math:`j \neq label_i`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function for i-th sample then becomes:

    .. math::
        \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, weight[label_i] * (\text{margin} - input_i[label_i] + input_i[j]))^p}{\text{C}}


    Parameters:

        p (int, Optional):The norm degree for pairwise distance. Default: :math:`1`.

        margin (float, Optional):Default: :math:`1`.

        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of shape (C,) and the data type is float32, float64.
                Default is ``'None'`` .

        reduction (str, optional): Indicate how to calculate the loss by batch_size,
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``

        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Call parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.

        label (Tensor): Label tensor, 0<= label < input.shape[1], the data type is int32 or int64.

    Shape:
        input: 2-D Tensor, the shape is [N, C], N is batch size and `C` means number of classes.

        label: 1-D Tensor, the shape is [N,].

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the label.

    Returns:
        A callable object of MultiMarginLoss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> label = paddle.to_tensor([0, 1, 2], dtype=paddle.int32)

            >>> multi_margin_loss = nn.MultiMarginLoss(reduction='mean')
            >>> loss = multi_margin_loss(input, label)
            >>> print(loss)
    """

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight=None,
        reduction="mean",
        name=None,
    ):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MultiMarginLoss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(reduction)
            )
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return F.multi_margin_loss(
            input,
            label,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
            name=self.name,
        )


class SoftMarginLoss(Layer):
    r"""

    Creates a criterion that measures a two-class soft margin loss between input predictions ``input``
    and target labels ``label`` . It can be described as:

    .. math::
        Out = log(1 + exp((-label * input)))

    Parameters:

        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.

        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
        - Input (Tensor): The input tensor with shape: ``[N, *]``,
          N is batch_size, `*` means any number of additional dimensions. The ``input`` ranges from -inf to inf
          Available dtype is float32, float64.
        - Label (Tensor): The target labels tensor with the same shape as
          ``input``. The target labels which values should be numbers -1 or 1.
          Available dtype is int32, int64, float32, float64.
        - Output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
          same as ``input`` , else the shape of output is [].

    Returns:
        A callable object of SoftMarginLoss.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[0.5, 0.6, 0.7],[0.3, 0.5, 0.2]], 'float32')
            >>> label = paddle.to_tensor([[1.0, -1.0, 1.0],[-1.0, 1.0, 1.0]], 'float32')
            >>> soft_margin_loss = paddle.nn.SoftMarginLoss()
            >>> output = soft_margin_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        0.64022040)

            >>> input_np = paddle.uniform(shape=(5, 5), min=0.1, max=0.8, dtype="float64")
            >>> label_np = paddle.randint(high=2, shape=(5, 5), dtype="int64")
            >>> label_np[label_np==0]=-1
            >>> input = paddle.to_tensor(input_np)
            >>> label = paddle.to_tensor(label_np)
            >>> soft_margin_loss = paddle.nn.SoftMarginLoss(reduction='none')
            >>> output = soft_margin_loss(input, label)
            >>> print(output)
            >>> # Tensor(shape=[5, 5], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[0.61739663, 0.51405668, 1.09346100, 0.42385561, 0.91602303],
            >>> #         [0.76997038, 1.01977148, 0.98971722, 1.13976032, 0.88152088],
            >>> #         [0.55476735, 1.10505384, 0.89923519, 0.45018155, 1.06587511],
            >>> #         [0.37998142, 0.48067240, 0.47791212, 0.55664053, 0.98581399],
            >>> #         [0.78571653, 0.59319711, 0.39701841, 0.76172109, 0.83781742]])

    """

    def __init__(self, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in SoftMarginLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )

        super().__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        out = paddle.nn.functional.soft_margin_loss(
            input, label, self.reduction, self.name
        )
        return out


class GaussianNLLLoss(Layer):
    r"""Create a callable object of 'GaussianNLLLoss' to calculate Gaussian negative log likelihood loss.

    This class create a callable object of Gaussian negative log likelihood loss among ``input``, ``variance`` and
    ``label``. Note that the ``label`` is treated as samples from Gaussian distributions.
    This class is used to train a neural network predicts
    the ``input`` and ``variance`` of a gaussian distribution that ``label`` are supposed to
    be coming from. This means ``input`` and ``variance`` should be functions(the neural network) of some inputs.

    For a ``label`` having Gaussian distribution with ``input`` and ``variance`` predicted by neural network
    the loss is calculated as follows:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{label}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`epsilon` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``variance`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``, means omit the constant term.
        epsilon (float, optional): value used to clamp ``variance`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - Input(Tensor): :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions. Available dtype is float32, float64.
        - Label(Tensor): :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting). Available dtype is float32, float64.
        - Variance(Tensor): :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting). Available dtype is float32, float64.
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Returns:
        A callable object of GaussianNLLLoss.

    Examples::
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> variance = paddle.ones([5, 2], dtype=paddle.float32)

            >>> gs_nll_loss = nn.GaussianNLLLoss(full=False, epsilon=1e-6, reduction='none')
            >>> loss = gs_nll_loss(input, label, variance)
            >>> print(loss)

    Note:
        The clamping of ``variance`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.
    """

    def __init__(self, full=False, epsilon=1e-6, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in GaussianNLLLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction
            )

        super().__init__()
        self.full = full
        self.epsilon = epsilon
        self.reduction = reduction
        self.name = name

    def forward(self, input, label, variance):
        out = F.gaussian_nll_loss(
            input,
            label,
            variance,
            self.full,
            self.epsilon,
            self.reduction,
            self.name,
        )
        return out


# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define the common classes to build a neural network
import paddle
from paddle import in_dynamic_mode

from .. import functional as F
from .layers import Layer

__all__ = []


def _npairs(x, n):
    if isinstance(x, (paddle.Tensor, list, tuple)):
        return x
    x = [x] * (n * 2)
    return x


class Identity(Layer):
    r"""

    A placeholder identity operator that is argument-insensitive. For each input :math:`X` ,
    the output :math:`Out` is:

    .. math::

        Out = X

    Parameters:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, n1, n2, ...]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, n1, n2, ...]` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input_tensor = paddle.randn(shape=[3, 2])
            >>> layer = paddle.nn.Identity()
            >>> out = layer(input_tensor)
            >>> # input_tensor: [[-0.32342386 -1.200079  ]
            >>> #                [ 0.7979031  -0.90978354]
            >>> #                [ 0.40597573  1.8095392 ]]
            >>> # out: [[-0.32342386 -1.200079  ]
            >>> #      [ 0.7979031  -0.90978354]
            >>> #      [ 0.40597573  1.8095392 ]]


    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Linear(Layer):
    r"""

    Fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None. If the Initializer of the
            param_attr is not set, the parameter is initialized with Xavier.
            For detailed information, please refer to paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` . Its data types are float16, float32, float64 ,The default is float32 .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` . The data type is the same as the input .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Define the linear layer.
            >>> weight_attr = paddle.ParamAttr(
            ...     name="weight",
            ...     initializer=paddle.nn.initializer.Constant(value=0.5))
            >>> bias_attr = paddle.ParamAttr(
            ...     name="bias",
            ...     initializer=paddle.nn.initializer.Constant(value=1.0))
            >>> linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
            >>> # linear.weight: [[0.5 0.5 0.5 0.5]
            >>> #                 [0.5 0.5 0.5 0.5]]
            >>> # linear.bias: [1. 1. 1. 1.]

            >>> x = paddle.randn((3, 2), dtype="float32")
            >>> # x: [[-0.32342386 -1.200079  ]
            >>> #     [ 0.7979031  -0.90978354]
            >>> #     [ 0.40597573  1.8095392 ]]
            >>> y = linear(x)
            >>> # y: [[0.23824859 0.23824859 0.23824859 0.23824859]
            >>> #     [0.9440598  0.9440598  0.9440598  0.9440598 ]
            >>> #     [2.1077576  2.1077576  2.1077576  2.1077576 ]]
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name

    def forward(self, input):
        out = F.linear(
            x=input, weight=self.weight, bias=self.bias, name=self.name
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str
        )


class LinearCompress(Layer):
    r"""

    Fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.

    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        weight_attr (ParamAttr, optional): The attribute for the weight of this layer.
            The default value is None. If the Initializer of the
            param_attr is not set, the parameter is initialized with Xavier.
            For detailed information, please refer to paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the bias of this layer.
            If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
        bits (int, optional): The attribute to set num of bits in quant during weight_only,
            it must be set as 8, default: 8.
        algo (str, optional): The  attribute to set algorithm of cpmoress, it must be set as 'weight_only'
            or 'llm.int8', default: weight_only.
        config (dict, optional): The parameter config for algorithm of cpmoress.
            For llm.int8, it should be set as {'threshold': 6.0}, default: {'threshold': 6.0}.

    Attribute:
        **weight** (Parameter): the learnable weight of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` . Its data types are float16.
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` . The data type is the same as the input .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Define the linear layer.
            >>> paddle.set_default_dtype('float16')
            >>> weight_attr = paddle.ParamAttr(
            ...     name="weight",
            ...     initializer=paddle.nn.initializer.Constant(value=0.5))
            >>> bias_attr = paddle.ParamAttr(
            ...     name="bias",
            ...     initializer=paddle.nn.initializer.Constant(value=1.0))
            >>> linear = paddle.nn.LinearCompress(128, 64, weight_attr=weight_attr, bias_attr=bias_attr, bits=8, algo='weight_only')
            >>> x = paddle.randn((3, 128), dtype="float16")
            >>> y = linear(x)
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
        bits=8,
        algo="weight_only",
        config={'threshold': 6.0},
    ):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.weight_scale = self.create_parameter(
            shape=[out_features],
            attr=None,
            dtype=self._dtype,
            is_bias=False,
        )
        self.is_weight_quanted = False
        self.name = (name,)
        self.bits = bits
        self.layout = algo
        self.algo = algo
        self.config = config

    def forward(self, input):
        if in_dynamic_mode():
            if not self.is_weight_quanted:
                weight_tensor, weight_scale_tensor = F.quant_for_compress(
                    self.weight, self.bits, self.layout
                )
                weight_attr = paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(weight_tensor)
                )
                weight_shape = (
                    [self.weight.shape[1], self.weight.shape[0]]
                    if self.bits == 8
                    else [self.weight.shape[1] / 2, self.weight.shape[0]]
                )
                self.weight = self.create_parameter(
                    shape=weight_shape,
                    attr=weight_attr,
                    dtype="int8",
                    is_bias=False,
                )
                weight_scale_attr = paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(
                        weight_scale_tensor
                    )
                )
                self.weight_scale = self.create_parameter(
                    shape=self.weight_scale.shape,
                    attr=weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )
                self.is_weight_quanted = True
            out = F.linear_compress(
                x=input,
                weight=self.weight,
                weight_scale=self.weight_scale,
                bias=self.bias,
                bits=self.bits,
                algo=self.algo,
                name=self.name,
                config=self.config,
            )
            return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}, algo={}'.format(
            self.weight.shape[0],
            self.weight.shape[1],
            self._dtype,
            name_str,
            self.algo,
        )


class Upsample(Layer):
    """
    This op resizes a batch of images.

    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    Where in_w is width of the input tensor, in_h is the height of the input tensor,
    in_d is the depth of the intput tensor.
    and the resizing only applies on the three dimensions(depth, height and width).

    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation

    Linear interpolation is the method of using a line connecting two known quantities
    to determine the value of an unknown quantity between the two known quantities.

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Area interpolation is to perform area interpolation
    in both the 3rd dimension(in height direction) , the 4th dimension(in width
    direction) and the 5th dimension(in depth direction) on input tensor. Set to
    area will directly call `paddle.nn.functional.adaptive_avg_pool1d` or
    `paddle.nn.functional.adaptive_avg_pool2d` or `paddle.nn.functional.adaptive_avg_pool3d`.

    Example:

    .. code-block:: text

        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}

        Nearest neighbor interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})
          else:
              align_corners = True
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    https://en.wikipedia.org/wiki/Linear_interpolation.
    For details of linear interpolation, please refer to Wikipedia:

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.

    Parameters:
        x (Tensor): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w)
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|Tensor|list|tuple|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`. Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearst', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand([2,3,6,10], dtype="float32")
            >>> upsample_out = paddle.nn.Upsample(size=[12,12])

            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            >>> # [2, 3, 12, 12]

    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=False,
        align_mode=0,
        data_format='NCHW',
        name=None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self):
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return '{}, mode={}, align_corners={}, align_mode={}, data_format={}{}'.format(
            main_str,
            self.mode,
            self.align_corners,
            self.align_mode,
            self.data_format,
            name_str,
        )


class UpsamplingNearest2D(Layer):
    """
    This op upsamples a batch of images, using nearest neighbours' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w),
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    Parameters:
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_data = paddle.rand(shape=(2,3,6,10)).astype("float32")
            >>> upsample_out  = paddle.nn.UpsamplingNearest2D(size=[12,12])
            >>> input = paddle.to_tensor(input_data)
            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            >>> # [2L, 3L, 12L, 12L]
    """

    def __init__(
        self, size=None, scale_factor=None, data_format='NCHW', name=None
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='nearest',
            align_corners=False,
            align_mode=0,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self):
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return '{}, data_format={}{}'.format(
            main_str, self.data_format, name_str
        )


class UpsamplingBilinear2D(Layer):
    """
    This op upsamples a batch of images, using bilinear' pixel values.
    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w),
    where in_w is width of the input tensor, in_h is the height of the input tensor.
    And the upsampling only applies on the two dimensions(height and width).
    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    Parameters:
        x (Tensor): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Tensor|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor.
             Default: None. If a list/tuple, each element can be an integer or a Tensor  of shape: [1].
             If a Tensor , its dimensions size should be a 1.
        scale_factor (float|int|list|tuple|Tensor|None): The multiplier for the input height or width. At
             least one of :attr:`size` or :attr:`scale_factor` must be set.
             And :attr:`size` has a higher priority than :attr:`scale_factor`.
             Has to match input size if it is either a list or a tuple or a Tensor.
             Default: None.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_data = paddle.rand(shape=(2,3,6,10)).astype("float32")
            >>> upsample_out  = paddle.nn.UpsamplingBilinear2D(size=[12,12])
            >>> input = paddle.to_tensor(input_data)
            >>> output = upsample_out(x=input)
            >>> print(output.shape)
            >>> # [2L, 3L, 12L, 12L]
    """

    def __init__(
        self, size=None, scale_factor=None, data_format='NCHW', name=None
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True,
            align_mode=0,
            data_format=self.data_format,
            name=self.name,
        )

        return out

    def extra_repr(self):
        if self.scale_factor is not None:
            main_str = f'scale_factor={self.scale_factor}'
        else:
            main_str = f'size={self.size}'
        name_str = f', name={self.name}' if self.name else ''
        return '{}, data_format={}{}'.format(
            main_str, self.data_format, name_str
        )


class Bilinear(Layer):
    r"""

    This layer performs bilinear on two inputs.

    .. math::

      out_{i} = x1 * W_{i} * {x2^\mathrm{T}}, i=0,1,...,outfeatures-1

      out = out + b

    In this formula:
     - :math:`x1`: the first input contains in1_features elements, shape is [batch_size, in1_features].
     - :math:`x2`: the second input contains in2_features elements, shape is [batch_size, in2_features].
     - :math:`W_{i}`: the i-th learned weight, shape is [in1_features, in2_features], and learned weight's shape is [out_features, in1_features, in2_features].
     - :math:`out_{i}`: the i-th element of out, shape is [batch_size], and out's shape is [batch_size, out_features].
     - :math:`b`: the learned bias, shape is [1, out_features].
     - :math:`x2^\mathrm{T}`: the transpose of :math:`x2`.

    Parameters:
       in1_features (int): The dimension of each first input(`x1`).
       in2_features (int): The dimension of each second input(`x2`).
       out_features (int): The dimension of output of this layer.
       weight_attr (ParamAttr, optional): The parameter attribute for the learnable w, parameters/weights of
       this layer. The default value is None.
       bias_attr (ParamAttr, optional): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. The default value is None.
       name (str, optional): The default value is None. Normally there is no need for user
           to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
       Tensor: A 2-D Tensor of shape [batch_size, out_features].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> layer1 = paddle.rand((5, 5)).astype('float32')
            >>> layer2 = paddle.rand((5, 4)).astype('float32')
            >>> bilinear = paddle.nn.Bilinear(
            ...     in1_features=5, in2_features=4, out_features=1000)
            >>> result = bilinear(layer1,layer2)    # result shape [5, 1000]

    """

    def __init__(
        self,
        in1_features,
        in2_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._name = name
        self._in1_features = in1_features
        self._in2_features = in2_features
        self._out_features = out_features
        self._dtype = self._helper.get_default_dtype()

        weight_shape = [
            self._out_features,
            self._in1_features,
            self._in2_features,
        ]
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=weight_shape,
            dtype=self._dtype,
            is_bias=False,
        )
        bias_shape = [1, self._out_features]
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=bias_shape,
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, x1, x2):
        return F.bilinear(x1, x2, self.weight, self.bias, self._name)

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return 'in1_features={}, in2_features={}, out_features={}, dtype={}{}'.format(
            self._in1_features,
            self._in2_features,
            self._out_features,
            self._dtype,
            name_str,
        )


class Dropout(Layer):
    r"""
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training as described in the paper:
    `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_
    The dropout operator randomly sets the outputs of some units to zero, while upscale others
    according to the given dropout probability.

    See :ref:`api_paddle_nn_functional_dropout` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5
        axis (int|list|tuple, optional): The axis along which the dropout is performed. Default: None.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train (default), upscale the output at training time

                                  - train: :math:`out = input \times \frac{mask}{(1.0 - p)}`
                                  - inference: :math:`out = input`

                               2. downscale_in_infer, downscale the output at inference

                                  - train: :math:`out = input \times mask`
                                  - inference: :math:`out = input \times (1.0 - p)`
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[1,2,3], [4,5,6]], dtype="float32")
            >>> m = paddle.nn.Dropout(p=0.5)

            >>> y_train = m(x)
            >>> print(y_train)
            >>> # Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[2., 0., 6.],
            >>> #         [0., 0., 0.]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            >>> # Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[1., 2., 3.],
            >>> #         [4., 5., 6.]])
    """

    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super().__init__()

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        out = F.dropout(
            input,
            p=self.p,
            axis=self.axis,
            training=self.training,
            mode=self.mode,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'p={}, axis={}, mode={}{}'.format(
            self.p, self.axis, self.mode, name_str
        )


class Dropout2D(Layer):
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW`). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout2D will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See :ref:`api_paddle_nn_functional_dropout2d` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float, optional): Probability of setting units to zero. Default: 0.5.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCHW` or `NHWC`. When it is `NCHW`, the data is stored in the order of: [batch_size, input_channels, input_height, input_width]. Default: `NCHW`.
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 4-D tensor.
        - output: 4-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([2, 2, 1, 3], dtype="float32")
            >>> print(x)
            >>> # Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[0.10052059, 0.93890846, 0.45351565]],
            >>> #          [[0.47507706, 0.45021373, 0.11331241]]],

            >>> #         [[[0.53358698, 0.97375143, 0.34997326]],
            >>> #          [[0.24758087, 0.52628899, 0.17970420]]]])

            >>> m = paddle.nn.Dropout2D(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            >>> # Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[0.        , 0.        , 0.        ]],
            >>> #          [[0.95015413, 0.90042746, 0.22662482]]],

            >>> #         [[[1.06717396, 1.94750285, 0.69994652]],
            >>> #          [[0.        , 0.        , 0.        ]]]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            >>> # Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[0.10052059, 0.93890846, 0.45351565]],
            >>> #          [[0.47507706, 0.45021373, 0.11331241]]],

            >>> #         [[[0.53358698, 0.97375143, 0.34997326]],
            >>> #          [[0.24758087, 0.52628899, 0.17970420]]]])
    """

    def __init__(self, p=0.5, data_format='NCHW', name=None):
        super().__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout2d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'p={}, data_format={}{}'.format(
            self.p, self.data_format, name_str
        )


class Dropout3D(Layer):
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout3D will help promote independence between feature maps as described in the paper:
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_

    See :ref:`api_paddle_nn_functional_dropout3d` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int, optional): Probability of setting units to zero. Default: 0.5.
        data_format (str, optional): Specify the data format of the input, and the data format of the output will be consistent with that of the input. An optional string from `NCDHW` or `NDHWC`. When it is `NCDHW`, the data is stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width]. Default: `NCDHW`.
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 5-D tensor.
        - output: 5-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(24, dtype="float32").reshape((1, 2, 2, 2, 3))
            >>> print(x)
            >>> # Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[[0. , 1. , 2. ],
            >>> #            [3. , 4. , 5. ]],
            >>> #           [[6. , 7. , 8. ],
            >>> #            [9. , 10., 11.]]],

            >>> #          [[[12., 13., 14.],
            >>> #            [15., 16., 17.]],
            >>> #           [[18., 19., 20.],
            >>> #            [21., 22., 23.]]]]])

            >>> m = paddle.nn.Dropout3D(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            >>> # Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[[0. , 2. , 4. ],
            >>> #            [6. , 8. , 10.]],
            >>> #           [[12., 14., 16.],
            >>> #            [18., 20., 22.]]],

            >>> #          [[[0. , 0. , 0. ],
            >>> #            [0. , 0. , 0. ]],
            >>> #           [[0. , 0. , 0. ],
            >>> #            [0. , 0. , 0. ]]]]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            >>> # Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[[[[0. , 1. , 2. ],
            >>> #            [3. , 4. , 5. ]],
            >>> #           [[6. , 7. , 8. ],
            >>> #            [9. , 10., 11.]]],

            >>> #          [[[12., 13., 14.],
            >>> #            [15., 16., 17.]],
            >>> #           [[18., 19., 20.],
            >>> #            [21., 22., 23.]]]]])
    """

    def __init__(self, p=0.5, data_format='NCDHW', name=None):
        super().__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout3d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'p={}, data_format={}{}'.format(
            self.p, self.data_format, name_str
        )


class AlphaDropout(Layer):
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with
    zero mean and unit standard deviation, the output of Alpha Dropout maintains the original mean and
    standard deviation of the input. Alpha Dropout fits well to SELU activate function by randomly setting
    activations to the negative saturation value.

    For more information, please refer to:
    `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[-1, 1], [-1, 1]], dtype="float32")
            >>> m = paddle.nn.AlphaDropout(p=0.5)
            >>> y_train = m(x)
            >>> print(y_train)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[-0.77919382,  1.66559887],
            >>> #         [-0.77919382, -0.77919382]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            >>> # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[-1.,  1.],
            >>> #         [-1.,  1.]])
    """

    def __init__(self, p=0.5, name=None):
        super().__init__()
        self.p = p
        self.name = name

    def forward(self, input):
        out = F.alpha_dropout(
            input, p=self.p, training=self.training, name=self.name
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}{name_str}'


class Pad1D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad1D`` class.
    Pad tensor according to ``pad``, ``mode`` and ``value``.
    If mode is ``reflect``, pad[0] and pad[1] must be no greater than width-1.

    Parameters:
        padding (Tensor|list[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in both dimensions. Else [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCL'``, ``'NLC'``. Specify the data format of the input data.
           Default: ``'NCL'``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 2, 3)
            >>> pad = [1, 2]
            >>> mode = "constant"
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad1D(padding=pad, mode=mode)
            >>> result = my_pad(data)
            >>> print(result)
            >>> # [[[0. 1. 2. 3. 0. 0.]
            >>> #   [0. 4. 5. 6. 0. 0.]]]
    """

    def __init__(
        self, padding, mode='constant', value=0.0, data_format="NCL", name=None
    ):
        super().__init__()
        self._pad = _npairs(padding, 1)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return 'padding={}, mode={}, value={}, data_format={}{}'.format(
            self._pad, self._mode, self._value, self._data_format, name_str
        )


class Pad2D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad2D`` class.
    Pad tensor according to ``pad``, ``mode`` and ``value``.
    If mode is ``'reflect'``, pad[0] and pad[1] must be no greater
    than width-1. The height dimension has the same condition.

    Parameters:
        padding (Tensor|list[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCHW'``, ``'NHWC'``. Specify the data format of the input data.
           Default: ``'NCHW'``.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2]
            >>> mode = "constant"
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad2D(padding=pad, mode=mode)
            >>> result = my_pad(data)
            >>> print(result)
            >>> # [[[[0. 0. 0. 0.]
            >>> #    [0. 1. 2. 3.]
            >>> #    [0. 4. 5. 6.]
            >>> #    [0. 0. 0. 0.]
            >>> #    [0. 0. 0. 0.]]]]
    """

    def __init__(
        self, padding, mode='constant', value=0.0, data_format="NCHW", name=None
    ):
        super().__init__()
        self._pad = _npairs(padding, 2)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return 'padding={}, mode={}, value={}, data_format={}{}'.format(
            self._pad, self._mode, self._value, self._data_format, name_str
        )


class ZeroPad2D(Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad2D`` class.
    Pads the input tensor boundaries with zero.

    Parameters:
        padding (Tensor | List[int] | int): The padding size with data type int. If is int, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions of input will be padded.
            The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x(Tensor): The input tensor of zeropad2d operator, which is a 4-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of zeropad2d operator, which is a 4-D tensor.
          The data type is same as input x.

    Examples:
        Examples are as follows.

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = paddle.to_tensor([1, 1, 2, 3])
            >>> pad = [1, 0, 1, 2]
            >>> data = paddle.arange(paddle.prod(input_shape), dtype="float32").reshape(input_shape) + 1

            >>> my_pad = nn.ZeroPad2D(padding=pad)
            >>> result = my_pad(data)

            >>> print(result)
            >>> # [[[[0. 0. 0. 0.]
            >>> #    [0. 1. 2. 3.]
            >>> #    [0. 4. 5. 6.]
            >>> #    [0. 0. 0. 0.]
            >>> #    [0. 0. 0. 0.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super().__init__()
        self._pad = _npairs(padding, 2)
        self._mode = 'constant'
        self._value = 0.0
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return 'padding={}, data_format={}{}'.format(
            self._pad, self._data_format, name_str
        )


class Pad3D(Layer):
    """
    This interface is used to construct a callable object of the ``Pad3D`` class.
    Pad tensor according to ``'pad'``, ``'mode'`` and ``'value'``.
    If mode is ``'reflect'``, pad[0] and pad[1] must be no greater
    than width-1. The height and depth dimension has the same condition.

    Parameters:
        padding (Tensor|list[int]|int): The padding size with data type ``'int'``. If is ``'int'``, use the
            same padding in all dimensions. Else [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default: ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas. Default is :math:`0.0`.
        data_format (str, optional): An string from: ``'NCDHW'``, ``'NDHWC'``. Specify the data format of the input data.
           Default:  ``'NCDHW'``。
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: ``'None'``.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input_shape = (1, 1, 1, 2, 3)
            >>> pad = [1, 0, 1, 2, 0, 0]
            >>> mode = "constant"
            >>> data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> my_pad = nn.Pad3D(padding=pad, mode=mode)
            >>> result = my_pad(data)
            >>> print(result)
            >>> # [[[[[0. 0. 0. 0.]
            >>> #     [0. 1. 2. 3.]
            >>> #     [0. 4. 5. 6.]
            >>> #     [0. 0. 0. 0.]
            >>> #     [0. 0. 0. 0.]]]]]
    """

    def __init__(
        self,
        padding,
        mode='constant',
        value=0.0,
        data_format="NCDHW",
        name=None,
    ):
        super().__init__()
        self._pad = _npairs(padding, 3)
        self._mode = mode
        self._value = value
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.pad(
            x,
            pad=self._pad,
            mode=self._mode,
            value=self._value,
            data_format=self._data_format,
            name=self._name,
        )

    def extra_repr(self):
        name_str = f', name={self._name}' if self._name else ''
        return 'padding={}, mode={}, value={}, data_format={}{}'.format(
            self._pad, self._mode, self._value, self._data_format, name_str
        )


class CosineSimilarity(Layer):
    """
    This interface is used to compute cosine similarity between x1 and x2 along axis.

    Parameters:
        axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
        eps(float): Small value to avoid division by zero. Default is 1e-8.
    Returns:
        None

    Examples:
        .. code-block:: text

            Case 0:
                x1 = [[0.8024077  0.9927354  0.27238318 0.8344984 ]
                     [0.48949873 0.5797396  0.65444374 0.66510963]
                     [0.1031398  0.9614342  0.08365563 0.6796464 ]
                     [0.10760343 0.7461209  0.7726148  0.5801006 ]]
                x2 = [[0.62913156 0.1536727  0.9847992  0.04591406]
                     [0.9098952  0.15715368 0.8671125  0.3156102 ]
                     [0.4427798  0.54136837 0.5276275  0.32394758]
                     [0.3769419  0.8535014  0.48041078 0.9256797 ]]
                axis = 1
                eps = 1e-8
                Out: [0.5275037  0.8368967  0.75037485 0.9245899]

    Code Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> x1 = paddle.to_tensor([[1., 2., 3.],
            ...                     [2., 3., 4.]], dtype="float32")
            >>> x2 = paddle.to_tensor([[8., 3., 3.],
            ...                     [2., 3., 4.]], dtype="float32")

            >>> cos_sim_func = nn.CosineSimilarity(axis=0)
            >>> result = cos_sim_func(x1, x2)
            >>> print(result)
            >>> # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            >>> #        [0.65079135, 0.98058069, 1.        ])
    """

    def __init__(self, axis=1, eps=1e-8):
        super().__init__()
        self._axis = axis
        self._eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, axis=self._axis, eps=self._eps)

    def extra_repr(self):
        return 'axis={_axis}, eps={_eps}'.format(**self.__dict__)


class Embedding(Layer):
    r"""

    Embedding Layer, used to construct a callable object of the ``Embedding`` class.
    For specific usage, refer to code examples. It implements the function of the Embedding Layer.
    This layer is used to lookup embeddings vector of ids provided by :attr:`x` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`num_embeddings` and :attr:`embedding_dim`.

    The shape of output Tensor is generated by appending an emb_size dimension to the
    last dimension of the input Tensor shape.

    Note:
        The id in :attr:`x` must satisfy :math:`0 =< id < num_embeddings` ,
        otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        x is a Tensor. padding_idx = -1
            x.data = [[1, 3], [2, 4], [4, 127]
            x.shape = [3, 2]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

    Parameters:
        num_embeddings (int): Just one element which indicate the size
            of the dictionary of embeddings.
        embedding_dim (int):  Just one element which indicate the size of each embedding vector respectively.
        padding_idx(int|long|None, optional): padding_idx needs to be in the interval [-num_embeddings, num_embeddings).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        sparse(bool, optional): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_paddle_optimizer_adadelta_Adadelta` , :ref:`api_paddle_optimizer_adamax_Adamax` , :ref:`api_paddle_optimizer_lamb_Lamb`.
            In these case, sparse must be False. Default: False.
        weight_attr(ParamAttr, optional): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example for details.
        name(str|None, optional): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.to_tensor([[0], [1], [3]], dtype="int64", stop_gradient=False)
            >>> embedding = paddle.nn.Embedding(4, 3, sparse=True)

            >>> w0 = paddle.to_tensor([[0., 0., 0.],
            ...                     [1., 1., 1.],
            ...                     [2., 2., 2.],
            ...                     [3., 3., 3.]], dtype="float32")
            >>> embedding.weight.set_value(w0)
            >>> print(embedding.weight)
            >>> # Tensor(shape=[4, 3], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            >>> #        [[0., 0., 0.],
            >>> #         [1., 1., 1.],
            >>> #         [2., 2., 2.],
            >>> #         [3., 3., 3.]])

            >>> adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
            >>> adam.clear_grad()


            >>> out = embedding(x)
            >>> print(out)
            >>> # Tensor(shape=[3, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            >>> #        [[[0., 0., 0.]],
            >>> #         [[1., 1., 1.]],
            >>> #         [[3., 3., 3.]]])

            >>> out.backward()
            >>> adam.step()

    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        sparse=False,
        weight_attr=None,
        name=None,
    ):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._sparse = sparse
        self._is_distributed = False
        self._padding_idx = padding_idx

        if self._num_embeddings <= 0:
            raise ValueError("num_embeddings must be gather than 0")

        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim must be gather than 0")

        padding_idx = (
            -1
            if padding_idx is None
            else padding_idx
            if padding_idx >= 0
            else (num_embeddings + padding_idx)
        )

        if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
            raise ValueError(
                "padding_idx must be within [-{}, {})".format(
                    num_embeddings, num_embeddings
                )
            )

        self._dtype = self._helper.get_default_dtype()
        self._size = [self._num_embeddings, self._embedding_dim]

        self._weight_attr = weight_attr
        self._remote_prefetch = False
        self._name = name
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False,
        )

        if in_dynamic_mode() and padding_idx != -1:
            with paddle.no_grad():
                self.weight[padding_idx] = 0.0

    def forward(self, x):
        return F.embedding(
            x,
            weight=self.weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name,
        )

    def extra_repr(self):
        main_str = '{_num_embeddings}, {_embedding_dim}'
        if self._padding_idx is not None:
            main_str += ', padding_idx={_padding_idx}'
        main_str += ', sparse={_sparse}'
        if self._name is not None:
            main_str += ', name={_name}'
        return main_str.format(**self.__dict__)


class Unfold(Layer):
    """
    Returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter sliding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    See ``paddle.nn.functional.unfold`` for more details.


    Parameters:
        kernel_sizes(int|list):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> x = paddle.randn((100,3,224,224))
            >>> unfold = nn.Unfold(kernel_sizes=[3, 3])
            >>> result = unfold(x)
            >>> print(result)
    """

    def __init__(
        self, kernel_sizes, dilations=1, paddings=0, strides=1, name=None
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.paddings = paddings
        self.strides = strides
        self.name = name

    def forward(self, input):
        return F.unfold(
            input,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
            name=self.name,
        )

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'kernel_size={}, dilation={}, padding={}, stride={}{}'.format(
            self.kernel_sizes,
            self.dilations,
            self.paddings,
            self.strides,
            name_str,
        )


class Fold(Layer):
    r"""

    Combines an array of sliding local blocks into a large containing
    tensor. also known as col2im when operated on batched 2D image tensor. Fold calculates each
    combined value in the resulting large tensor by summing all values from all containing blocks.


    For each input :math:`x` with shape [N, C_in , L], the output shape [N, C_out, H_out, W_out]
    can be calculated as following.

    .. math::

        H_{out} &= output\_size[0] \\
        W_{out} &= output\_size[1] \\
        C_{out} &= \frac{C_{in}}{kernel\_sizes[0]\times kernel\_sizes[1]} \\

    Parameters:
        output_sizes(list):       The size of output size, should be [output_size_h, output_size_w]
                                  or an interger o treated as [o, o].
        kernel_sizes(int|list|tuple):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list|tuple, optional):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list|tuple, optional):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list|tuple, optional):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        The tensor formed by combining a group of sliding local blocks
        The output shape is [N, Cout, H, W] as decriabled above.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> x = paddle.randn([2,3*2*2,12])
            >>> fold = nn.Fold(output_sizes=[4, 5], kernel_sizes=2)
            >>> y = fold(x)
            >>> # y.shape = [2,3,4,5]
   """

    def __init__(
        self,
        output_sizes,
        kernel_sizes,
        dilations=1,
        paddings=0,
        strides=1,
        name=None,
    ):
        super().__init__()

        self.output_sizes = output_sizes
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.paddings = paddings
        self.strides = strides
        self.name = name

    def forward(self, input):
        return F.fold(
            input,
            output_sizes=self.output_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
            name=self.name,
        )

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return 'kernel_size={}, dilation={}, padding={}, stride={}{}'.format(
            self.kernel_sizes,
            self.dilations,
            self.paddings,
            self.strides,
            name_str,
        )


class Flatten(Layer):
    """
    This interface is used to construct a callable object of the ``FLatten`` class.
    For more details, refer to code examples.
    It implements flatten a contiguous range of dims into a tensor.

    Parameters:
        start_axis(int): first dim to flatten (default = 1)
        stop_axis(int): last dim to flatten (default = -1).

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> inp = paddle.ones([5, 2, 3, 4]).astype('float32')
            >>> flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
            >>> y = flatten(inp)
            >>> # y.shape = [5, 6, 4]

    """

    def __init__(self, start_axis=1, stop_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.stop_axis = stop_axis

    def forward(self, input):
        out = paddle.flatten(
            input, start_axis=self.start_axis, stop_axis=self.stop_axis
        )
        return out


class Unflatten(Layer):
    """
    This interface is used to construct a callable object of the ``Unflatten`` class.
    For more details, refer to code examples.
    It a certain dimension of the input x Tensor into a desired shape.

    Parameters:
        axis (int): :attr:`axis` to be unflattened, specified as an index into `x.shape`.
        shape (list|tuple|Tensor): Unflatten :attr:`shape` on the specified :attr:`axis`. At most one dimension of the target :attr:`shape` can be -1.
            If the input :attr:`shape` does not contain -1 , the product of all elements in ``shape`` should be equal to ``x.shape[axis]``.
            The data type is `int` . If :attr:`shape` is a list or tuple, the elements of it should be integers or Tensors with shape [].
            If :attr:`shape` is an Tensor, it should be an 1-D Tensor.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn(shape=[4, 6, 8])
            >>> shape = [2, 3]
            >>> axis = 1
            >>> unflatten = paddle.nn.Unflatten(axis, shape)
            >>> res = unflatten(x)
            >>> print(res.shape)
            >>> # [4, 2, 3, 8]

    """

    def __init__(self, axis, shape, name=None):
        super().__init__()
        self.axis = axis
        self.shape = shape
        self.name = name

    def forward(self, input):
        out = paddle.unflatten(
            input, axis=self.axis, shape=self.shape, name=self.name
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return f'axis={self.axis}, shape={self.shape}{name_str}'
