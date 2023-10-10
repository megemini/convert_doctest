def gpu():
    """
    
    $ convert-doctest doctest test/test_capacity.py
    Result to `SKIPPED`

    $ convert-doctest doctest test/test_capacity.py -c cpu gpu
    Result to `SUCCESS`

    Examples:
        .. code-block:: python

        >>> # doctest: +REQUIRES(env:GPU)
        >>> print('gpu')
        gpu
    
    """

def gpu_xpu():
    """
    
    $ convert-doctest doctest test/test_capacity.py -c cpu gpu
    Result to `SKIPPED`

    $ convert-doctest doctest test/test_capacity.py -c cpu gpu xpu
    Result to `SUCCESS`

    Examples:
        .. code-block:: python

        >>> # doctest: +REQUIRES(env:GPU, env:XPU)
        >>> print('gpu xpu')
        gpu xpu
    
    """