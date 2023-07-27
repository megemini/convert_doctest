def test(a):
    """this is docstring...

    Examples:

        .. code-block:: python

            this is a test...

            >>> # doctest: +SKIP
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
