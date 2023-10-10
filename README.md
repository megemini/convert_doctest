# convert_doctest

## 特性
#### 示例转换
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

#### 文件看门狗
监控修改的文件，并生成临时文件，以方便使用 `xdoctest` 进行检查。

#### 示例检查
使用 `xdoctest` 或 `convert-doctest` 对上述转换的示例代码进行检查。

#### 旧格式检查
检查文件是否仍然使用旧格式（不使用 `>>>`）的示例代码。

## 安装

```bash
$ pip install git+https://github.com/megemini/convert_doctest@main
```

## 使用方法

#### 方法一 : 批量处理

首先使用 `convert-doctest` 将旧格式转换为新格式：

```shell
$ convert-doctest convert source_file.py --target target_file.py
```

如果在同一个文件修改：

```shell
$ convert-doctest convert source_file.py
```

`source_file.py` 和 `target_file.py` 是脚本转换的示例～

转换之后运行 `watch-docstring` 监控该文件的修改（修改源文件，生成的文件会自动更新）

```bash
$ watch-docstring target_file.py
```

此时如果你修改文件的话，会生成一个 `xdoctest_test` 目录，里面包含了全部提取得到的 docstring，此时你只需要新开一个终端运行如下命令即可测试全部示例代码。

```bash
$ xdoctest \
  --debug --options "+IGNORE_WHITESPACE" --style "freeform" \
  --global-exec "import paddle\npaddle.device.set_device('cpu')" \
  xdoctest_test
```

#### 方法二 : 单个处理

首先使用 `convert-doctest` 将旧格式转换为新格式：

```shell
$ convert-doctest convert source_file.py --target target_file.py
```

如果在同一个文件修改：

```shell
$ convert-doctest convert source_file.py
```

对单个文件进行检查：

```shell
$ convert-doctest --debug doctest target_file.py
```

检查时添加运行时环境：

```shell
$ convert-doctest doctest target_file.py --capacity cpu gpu
```

或

```shell
$ convert-doctest doctest target_file.py -c cpu gpu
```

检查时添加 `xdoctester` 参数：

```shell
$ convert-doctest doctest target_file.py -c cpu gpu xpu --kwargs "{'patch_float_precision':6}"
```

#### 旧格式检查

```shell
$ convert-doctest nocodes target_file.py
```

如果检查出文件中仍有使用旧格式示例代码的情况，将会逐一列出。

## 注意事项

这个脚本只是辅助进行文件的转换，而且脚本依赖缩进的正确性，转换后仍存在较多问题！

可以转换后把代码复制到 ipython 中执行，然后把执行结果复制进去，效率至少提升一倍！

：）
