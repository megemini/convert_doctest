# from Paddle tools

import collections
import time
import typing
import functools
import logging
import os
import platform
import re
import sys
import time
import typing

import xdoctest

logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[
        0
    ]  # we assume the first handler is the one we want to configure
else:
    console = logging.StreamHandler(stream=sys.stderr)
    logger.addHandler(console)
console.setFormatter(logging.Formatter("%(message)s"))


TEST_TIMEOUT = 10

XDOCTEST_CONFIG = {
    "global_exec": r"\n".join(
        [
            "import paddle",
            "paddle.device.set_device('cpu')",
        ]
    ),
    "analysis": "auto",
    "options": "+IGNORE_WHITESPACE",
}


def init_logger(debug=True, log_file=None):
    """
    init logger level and file handler
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if log_file is not None:
        logfHandler = logging.FileHandler(log_file)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(logfHandler)


TestResult = collections.namedtuple(
    "TestResult",
    (
        "name",
        "nocode",
        "passed",
        "skipped",
        "failed",
        "time",
        "test_msg",
        "extra_info",
    ),
    defaults=(None, False, False, False, False, -1, "", None),
)


class DocTester:
    """A DocTester can be used to test the codeblock from the API's docstring.

    Attributes:

        style(str): `style` should be in {'google', 'freeform'}.
            `google`, codeblock in `Example(s):` section of docstring.
            `freeform`, all codeblocks in docstring wrapped with PS1(>>> ) and PS2(... ).
            **CAUTION** no matter `.. code-block:: python` used or not,
                the docstring in PS1(>>> ) and PS2(... ) should be considered as codeblock.
        target(str): `target` should be in {'docstring', 'codeblock'}.
            `docstring`, the test target is a docstring with optional description, `Args:`, `Returns:`, `Examples:` and so on.
            `codeblock`, the codeblock extracted by `extract_code_blocks_from_docstr` from the docstring, and the pure codeblock is the docstring to test.
                If we use `.. code-block:: python` wrapping the codeblock, the target should be `codeblock` instead of `docstring`.
                Because the `doctest` and `xdoctest` do NOT care the `.. code-block:: python` directive.
                If the `style` is set to `google` and `target` is set to `codeblock`, we should implement/overwrite `ensemble_docstring` method,
                where ensemble the codeblock into a docstring with a `Examples:` and some indents as least.
        directives(list[str]): `DocTester` hold the default directives, we can/should replace them with method `convert_directive`.
    """

    style = 'google'
    target = 'docstring'
    directives = [
        "# doctest: +SKIP",
        "# doctest: +REQUIRES(env:CPU)",
        "# doctest: +REQUIRES(env:GPU)",
        "# doctest: +REQUIRES(env:XPU)",
        "# doctest: +REQUIRES(env:DISTRIBUTED)",
        "# doctest: +REQUIRES(env:GPU, env:XPU)",
    ]

    def ensemble_docstring(self, codeblock: str) -> str:
        """Ensemble a cleaned codeblock into a docstring.

        For example, we can add `Example:` before the code block and some indents, which makes it a `google` style docstring.
        Otherwise, a codeblock is just a `freeform` style docstring.

        Args:
            codeblock(str): a str of codeblock and its outputs.

        Returns:
            a docstring for test.
        """
        if self.style == 'google':
            return 'Examples:\n' + '\n'.join(
                ['    ' + line for line in codeblock.splitlines()]
            )

        return codeblock

    def convert_directive(self, docstring: str) -> str:
        """Convert the standard directive from default DocTester into the doctester's style:

        For example:
        From: # doctest: +SKIP
        To: # xdoctest: +SKIP

        Args:
            docstring(str): the raw docstring

        Returns:
            a docstring with directives converted.
        """
        return docstring

    def prepare(self, test_capacity: set) -> None:
        """Something before run the test.

        Xdoctest need to set the `os.environ` according to the test capacity,
        which `+REQUIRES` used to match the test required environment.

        Legacy sample code processor do NOT need.

        Args:
            test_capacity(set): the test capacity, like `cpu`, `gpu` and so on.
        """
        pass

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Extract codeblocks from docstring, and run the test.
        Run only one docstring at a time.

        Args:
            api_name(str): api name
            docstring(str): docstring.

        Returns:
            list[TestResult]: test results. because one docstring may extract more than one code examples, so return a list.
        """
        raise NotImplementedError

    def print_summary(
        self, test_results: typing.List[TestResult], whl_error: typing.List[str]
    ) -> None:
        """Post process test results and print test summary.

        There are some `required not match` in legacy test processor, but NOT exist in Xdoctest.
        When using the legacy processor, we can set test result to `skipped=True` and store the `not match` information in `extra_info`,
        then logging the `not match` in `print_summary`.

        Args:
            test_results(list[TestResult]): test results generated from doctester.
            whl_error(list[str]): wheel error when we extract apis from module.
        """
        pass


class Xdoctester(DocTester):
    """A Xdoctest doctester."""

    def __init__(
        self,
        debug=False,
        style='freeform',
        target='codeblock',
        mode='native',
        verbose=2,
        **config,
    ):
        self.debug = debug

        self.style = style
        self.target = target
        self.mode = mode
        self.verbose = verbose
        self.config = {**XDOCTEST_CONFIG, **(config or {})}

        # patch xdoctest global_state
        from xdoctest import global_state

        _debug_xdoctest = debug and verbose > 2
        global_state.DEBUG = _debug_xdoctest
        global_state.DEBUG_PARSER = (
            global_state.DEBUG_PARSER and _debug_xdoctest
        )
        global_state.DEBUG_CORE = global_state.DEBUG_CORE and _debug_xdoctest
        global_state.DEBUG_RUNNER = (
            global_state.DEBUG_RUNNER and _debug_xdoctest
        )
        global_state.DEBUG_DOCTEST = (
            global_state.DEBUG_DOCTEST and _debug_xdoctest
        )

        self.docstring_parser = functools.partial(
            xdoctest.core.parse_docstr_examples, style=self.style
        )

        self.directive_pattern = re.compile(
            r"""
            (?<=(\#\s{1}))  # positive lookbehind, directive begins
            (doctest)   # directive prefix, which should be replaced
            (?= # positive lookahead, directive content
                (
                    :\s+
                    [\+\-]
                    (REQUIRES|SKIP)
                    (\((env\s*:\s*(CPU|GPU|XPU|DISTRIBUTED)\s*,?\s*)+\))?
                )
                \s*\n+
            )""",
            re.X,
        )
        self.directive_prefix = 'xdoctest'

    def convert_directive(self, docstring: str) -> str:
        """Replace directive prefix with xdoctest"""
        return self.directive_pattern.sub(self.directive_prefix, docstring)

    def prepare(self, test_capacity: set):
        """Set environs for xdoctest directive.
        The keys in environs, which also used in `# xdoctest: +REQUIRES(env:XX)`, should be UPPER case.

        If `test_capacity = {"cpu"}`, then we set:

            - `os.environ["CPU"] = "True"`

        which makes this SKIPPED:

            - # xdoctest: +REQUIRES(env:GPU)

        If `test_capacity = {"cpu", "gpu"}`, then we set:

            - `os.environ["CPU"] = "True"`
            - `os.environ["GPU"] = "True"`

        which makes this SUCCESS:

            - # xdoctest: +REQUIRES(env:GPU)
        """
        logger.info("Set xdoctest environ ...")
        for capacity in test_capacity:
            key = capacity.upper()
            os.environ[key] = "True"
            logger.info("Environ: %s , set to True.", key)

        logger.info("API check using Xdoctest prepared!-- Example Code")
        logger.info("running under python %s", platform.python_version())
        logger.info("running under xdoctest %s", xdoctest.__version__)

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Run the xdoctest with a docstring."""
        examples_to_test, examples_nocode = self._extract_examples(
            api_name, docstring
        )
        return self._execute_xdoctest(examples_to_test, examples_nocode)

    def _extract_examples(self, api_name, docstring):
        """Extract code block examples from docstring."""
        examples_to_test = {}
        examples_nocode = {}
        for example_idx, example in enumerate(
            self.docstring_parser(docstr=docstring, callname=api_name)
        ):
            example.mode = self.mode
            example.config.update(self.config)
            example_key = f"{api_name}_{example_idx}"

            # check whether there are some parts parsed by xdoctest
            if not example._parts:
                examples_nocode[example_key] = example
                continue

            examples_to_test[example_key] = example

        if not examples_nocode and not examples_to_test:
            examples_nocode[api_name] = api_name

        return examples_to_test, examples_nocode

    def _execute_xdoctest(self, examples_to_test, examples_nocode):
        """Run xdoctest for each example"""
        test_results = []
        for _, example in examples_to_test.items():
            start_time = time.time()
            result = example.run(verbose=self.verbose, on_error='return')
            end_time = time.time()

            test_results.append(
                TestResult(
                    name=str(example),
                    passed=result['passed'],
                    skipped=result['skipped'],
                    failed=result['failed'],
                    test_msg=result['exc_info'],
                    time=end_time - start_time,
                )
            )

        for _, example in examples_nocode.items():
            test_results.append(TestResult(name=str(example), nocode=True))

        return test_results

    def print_summary(self, test_results, whl_error=None):
        summary_success = []
        summary_failed = []
        summary_skiptest = []
        summary_nocodes = []

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        # logger.addHandler(stdout_handler)
        logger.info("----------------End of the Check--------------------")
        if whl_error is not None:
            logger.info("%s is not in whl.", whl_error)
            logger.info("")
            logger.info("Please check the whl package and API_PR.spec!")
            logger.info(
                "You can follow these steps in order to generate API.spec:"
            )
            logger.info("1. cd ${paddle_path}, compile paddle;")
            logger.info("2. pip install build/python/dist/(build whl package);")
            logger.info(
                "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
            )
            for test_result in test_results:
                if test_result.failed:
                    logger.info(
                        "In addition, mistakes found in sample codes: %s",
                        test_result.name,
                    )
            logger.info("----------------------------------------------------")
            sys.exit(1)
        else:
            timeovered_test = {}
            for test_result in test_results:
                if not test_result.nocode:
                    if test_result.passed:
                        summary_success.append(test_result.name)

                    if test_result.skipped:
                        summary_skiptest.append(test_result.name)

                    if test_result.failed:
                        summary_failed.append(test_result.name)

                    if test_result.time > TEST_TIMEOUT:
                        timeovered_test[test_result.name] = test_result.time
                else:
                    summary_nocodes.append(test_result.name)

            if len(timeovered_test):
                logger.info(
                    "%d sample codes ran time over 10s", len(timeovered_test)
                )
                if self.debug:
                    for k, v in timeovered_test.items():
                        logger.info(f'{k} - {v}s')
            if len(summary_success):
                logger.info("%d sample codes ran success", len(summary_success))
                logger.info('\n'.join(summary_success))

            if len(summary_skiptest):
                logger.info("%d sample codes skipped", len(summary_skiptest))
                logger.info('\n'.join(summary_skiptest))

            if len(summary_nocodes):
                logger.info(
                    "%d apis could not run test or don't have sample codes", len(summary_nocodes)
                )
                logger.info('\n'.join(summary_nocodes))

            if len(summary_failed):
                logger.info("%d sample codes ran failed", len(summary_failed))
                logger.info('\n'.join(summary_failed))
                logger.info(
                    "Mistakes found in sample codes. Please recheck the sample codes."
                )
                sys.exit(1)

        logger.info("Sample code check is successful!")
