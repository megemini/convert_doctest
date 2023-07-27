# convert_doctest

把没有 `>>> ` 的 `.. code-block:: python` 中的代码进行封装：

- 对于单独的代码行添加 `>>> `
- 对于继续的代码行添加 `... `
- 如果缩进不是 `4` 的整数倍，则格式化为 `4` 的整数倍
- 把多个空格的
  ```
  ..  code-block:: python
  ```
  转为
  ```
  .. code-block:: python
  ```

## 安装

```bash
$ pip install git+https://github.com/megemini/convert_doctest@main
```

## 使用方法

首先使用 `convert-doctest` 将旧格式转换为新格式：

```shell
$ convert-doctest source_file.py target_file.py
```

如果在同一个文件修改：

```shell
$ convert-doctest source_file.py
```

`source_file.py` 和 `target_file.py` 是脚本转换的示例～

之后运行 `watch-docstring` 监控该文件的修改（修改源文件，生成的文件会自动更新）

```bash
$ watch-docstring target_file.py
```

此时如果你修改文件的话，会生成一个 `xdoctest_test` 目录，里面包含了全部提取得到的 docstring，此时你只需要新开一个终端运行如下命令即可测试全部示例代码

```bash
$ xdoctest --global-exec "import paddle\npaddle.device.set_device('cpu')" xdoctest_test
```

## 注意事项

这个脚本只是辅助进行文件的转换，而且脚本依赖缩进的正确性，转换后仍存在较多问题！

可以转换后把代码复制到 ipython 中执行，然后把执行结果复制进去，效率至少提升一倍！

：）
