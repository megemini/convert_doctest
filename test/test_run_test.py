def test(a):
    """this is docstring...

    Examples:

        .. code-block:: python

            this is a test...

            >>> a = 3
            >>> print(a)
            3

            >>> class A:
            ...     def a(self, a):
            ...         return a+1
            ...
            ...     def b(self, x):
            ...         return x
            ...
            >>> print(A().a(a))
            4

            >>> print(A().b(a))
            3

    """
    pass

def test():
    """
    >>> def func(x):
    ...     if paddle.mean(x) < 0:
    ...         x_v = x - 1
    ...     else:
    ...         x_v = x + 1
    ...     return x_v
    """
    pass

