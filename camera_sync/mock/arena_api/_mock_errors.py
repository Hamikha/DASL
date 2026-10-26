"""Exception types raised by the mock.

The real arena_api's exception classes are not documented publicly, so the mock derives
from Python built-ins and capture code should catch those built-ins (or plain Exception):
  * value out of range / invalid enum entry / wrong type   -> ValueError
  * node not writable / not readable (e.g. while streaming) -> RuntimeError
  * get_buffer timeout                                      -> TimeoutError
  * nm['Missing']                                           -> KeyError
"""


class MockOutOfRangeError(ValueError):
    pass


class MockAccessError(RuntimeError):
    pass


class MockTimeoutError(TimeoutError):
    pass


class MockStreamError(RuntimeError):
    pass
