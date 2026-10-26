"""GenICam-like node objects and node maps for the mock.

Only the behaviour capture scripts depend on is modelled:
  * typed nodes (Integer / Float / Boolean / Enumeration / String / Command) with
    .name .value .is_readable .is_writable .min .max .inc .unit .enumentry_names .execute()
  * range / increment / entry validation -> ValueError
  * access control (read-only nodes, nodes locked while the stream runs) -> RuntimeError
  * "selected" nodes: one value per selector entry (TriggerMode per TriggerSelector,
    ChunkEnable per ChunkSelector, ...), exactly like GenICam selectors
  * computed (live) values, e.g. PtpStatus or ExposureTime under auto exposure
"""
from __future__ import annotations

import math
import operator
import threading
from typing import Any, Callable, Iterable

from ._mock_errors import MockAccessError, MockOutOfRangeError
from .enums import AccessMode, InterfaceType

BoolOrFn = bool | Callable[[], bool]


def _eval(x: Any) -> Any:
    return x() if callable(x) else x


class NodeContext:
    """What nodes need from their owner: a lock and 'is the stream running?'."""

    def __init__(self, lock: threading.RLock | None = None,
                 streaming: Callable[[], bool] | None = None) -> None:
        self.lock = lock or threading.RLock()
        self.streaming = streaming or (lambda: False)


class Node:
    interface_type = InterfaceType.BASE

    def __init__(self, name: str, ctx: NodeContext, *, value: Any = None,
                 readable: BoolOrFn = True, writable: BoolOrFn = True,
                 lock_while_streaming: bool = False,
                 getter: Callable[[], Any] | None = None,
                 on_change: Callable[["Node", Any, Any], None] | None = None,
                 selector: str | None = None, nodemap: "NodeMap | None" = None,
                 unit: str = "", description: str = "", display_name: str | None = None
                 ) -> None:
        self._name = name
        self._ctx = ctx
        self._readable = readable
        self._writable = writable
        self._lock_while_streaming = lock_while_streaming
        self._getter = getter          # computed value; overrides the stored value
        self._on_change = on_change    # side effects after a successful write
        self._selector = selector      # name of the selector node, if any
        self._nodemap = nodemap
        self._values: dict[Any, Any] = {}
        self._default = value
        self.unit = unit
        self.description = description
        self.display_name = display_name or name

    # -------------------------------------------------------------- identity / access
    @property
    def name(self) -> str:
        return self._name

    @property
    def is_readable(self) -> bool:
        return bool(_eval(self._readable))

    @property
    def is_writable(self) -> bool:
        if self._lock_while_streaming and self._ctx.streaming():
            return False
        return bool(_eval(self._writable))

    @property
    def access_mode(self) -> AccessMode:
        r, w = self.is_readable, self.is_writable
        if r and w:
            return AccessMode.RW
        if r:
            return AccessMode.RO
        if w:
            return AccessMode.WO
        return AccessMode.NA

    def __repr__(self) -> str:
        try:
            v = self.value if self.is_readable else "<not readable>"
        except Exception as exc:  # pragma: no cover - repr must not raise
            v = f"<{exc}>"
        return f"<{type(self).__name__} {self._name}={v!r}>"

    # -------------------------------------------------------------- selector support
    def _key(self) -> Any:
        if self._selector is None or self._nodemap is None:
            return None
        return self._nodemap[self._selector].value

    def _stored(self) -> Any:
        return self._values.get(self._key(), self._default)

    def raw_get(self, selector_value: Any = None) -> Any:
        """Mock-internal: STORED value for a selector entry (no access check, no getter)."""
        if self._selector is None:
            return self._values.get(None, self._default)
        return self._values.get(selector_value, self._default)

    def raw_set(self, value: Any, selector_value: Any = None) -> None:
        """Mock-internal: bypass access checks (firmware writing its own registers)."""
        with self._ctx.lock:
            self._values[selector_value if self._selector else None] = value

    # -------------------------------------------------------------- value
    @property
    def value(self) -> Any:
        with self._ctx.lock:
            if not self.is_readable:
                raise MockAccessError(f"node {self._name} is not readable")
            if self._getter is not None:
                return self._getter()
            return self._stored()

    @value.setter
    def value(self, new: Any) -> None:
        with self._ctx.lock:
            if not self.is_writable:
                why = " while streaming" if (self._lock_while_streaming
                                             and self._ctx.streaming()) else ""
                raise MockAccessError(f"node {self._name} is not writable{why}")
            new = self._validate(new)
            if getattr(self, "_stuck", False):
                return            # mock fault knob stuck_nodes: write accepted, no effect
            key = self._key()
            old = self._values.get(key, self._default)
            self._values[key] = new
            if self._on_change is not None:
                self._on_change(self, old, new)

    def _validate(self, new: Any) -> Any:  # overridden
        return new


class _NumericNode(Node):
    def __init__(self, name: str, ctx: NodeContext, *, min: Any = None, max: Any = None,
                 inc: Any = None, **kw: Any) -> None:
        super().__init__(name, ctx, **kw)
        self._min, self._max, self._inc = min, max, inc

    @property
    def min(self) -> Any:
        return _eval(self._min)

    @property
    def max(self) -> Any:
        return _eval(self._max)

    @property
    def inc(self) -> Any:
        return _eval(self._inc)

    def _check_range(self, v: Any) -> None:
        lo, hi = self.min, self.max
        if lo is not None and v < lo:
            raise MockOutOfRangeError(f"{self._name}: {v} < min {lo}")
        if hi is not None and v > hi:
            raise MockOutOfRangeError(f"{self._name}: {v} > max {hi}")


class IntegerNode(_NumericNode):
    interface_type = InterfaceType.INTEGER

    def _validate(self, new: Any) -> int:
        if isinstance(new, bool):
            raise MockOutOfRangeError(f"{self._name}: expected int, got bool")
        if isinstance(new, float):
            if not new.is_integer():
                raise MockOutOfRangeError(f"{self._name}: expected int, got {new!r}")
            new = int(new)
        try:
            v = operator.index(new)
        except TypeError:
            raise MockOutOfRangeError(f"{self._name}: expected int, got {type(new).__name__}")
        self._check_range(v)
        inc, lo = self.inc, self.min
        if inc and inc > 1 and lo is not None and (v - lo) % inc:
            raise MockOutOfRangeError(f"{self._name}: {v} violates increment {inc}")
        return v


class FloatNode(_NumericNode):
    interface_type = InterfaceType.FLOAT

    def _validate(self, new: Any) -> float:
        if isinstance(new, bool) or not isinstance(new, (int, float)) and not hasattr(new, "__float__"):
            raise MockOutOfRangeError(f"{self._name}: expected float, got {type(new).__name__}")
        v = float(new)
        if math.isnan(v):
            raise MockOutOfRangeError(f"{self._name}: NaN")
        self._check_range(v)
        return v


class BooleanNode(Node):
    interface_type = InterfaceType.BOOLEAN

    def _validate(self, new: Any) -> bool:
        if isinstance(new, bool) or type(new).__name__ == "bool_":
            return bool(new)
        if isinstance(new, int) and new in (0, 1):
            return bool(new)
        raise MockOutOfRangeError(f"{self._name}: expected bool, got {new!r}")


class EnumerationNode(Node):
    interface_type = InterfaceType.ENUMERATION

    def __init__(self, name: str, ctx: NodeContext, *, entries: Iterable[str] | Callable,
                 **kw: Any) -> None:
        super().__init__(name, ctx, **kw)
        self._entries = entries if callable(entries) else list(entries)

    @property
    def enumentry_names(self) -> list[str]:
        return list(_eval(self._entries))

    def _validate(self, new: Any) -> str:
        # The real arena_api accepts enum members for enum nodes (the user's capture.py
        # writes PixelFormat.Coord3D_ABCY16); the mock maps them to their entry name.
        if not isinstance(new, str) and isinstance(getattr(new, "name", None), str):
            new = new.name
        if not isinstance(new, str):
            raise MockOutOfRangeError(f"{self._name}: expected str entry, got {new!r}")
        if new not in self.enumentry_names:
            raise MockOutOfRangeError(
                f"{self._name}: {new!r} is not an entry ({self.enumentry_names})")
        return new


class StringNode(Node):
    interface_type = InterfaceType.STRING

    def _validate(self, new: Any) -> str:
        if not isinstance(new, str):
            raise MockOutOfRangeError(f"{self._name}: expected str, got {new!r}")
        return new


class CommandNode(Node):
    interface_type = InterfaceType.COMMAND

    def __init__(self, name: str, ctx: NodeContext, *, action: Callable[[], None],
                 **kw: Any) -> None:
        super().__init__(name, ctx, **kw)
        self._action = action

    def execute(self) -> None:
        with self._ctx.lock:
            if not self.is_writable:
                raise MockAccessError(f"command {self._name} is not executable now")
        # Run the action OUTSIDE the node lock where possible: actions may touch other
        # cameras (ActionCommandFireCommand) and must not hold this camera's lock.
        self._action()

    def is_done(self) -> bool:
        return True

    @property
    def value(self) -> Any:  # GenICam commands have no readable value
        raise MockAccessError(f"command {self._name} has no value; use execute()")

    @value.setter
    def value(self, new: Any) -> None:
        raise MockAccessError(f"command {self._name} has no value; use execute()")


class NodeMap:
    """Dict-like node map. ``nm['X']`` raises KeyError when missing; ``get_node`` returns
    None (single name) or a dict (list of names, missing ones map to None)."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._nodes: dict[str, Node] = {}

    def _add(self, node: Node) -> Node:
        node._nodemap = self
        self._nodes[node.name] = node
        return node

    def _remove(self, name: str) -> None:
        self._nodes.pop(name, None)

    def __getitem__(self, name: str) -> Node:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(f"node {name!r} not found in {self.title}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    def __iter__(self):
        return iter(list(self._nodes.values()))

    def __len__(self) -> int:
        return len(self._nodes)

    def get_node(self, names: str | Iterable[str]) -> Node | dict[str, Node | None] | None:
        if isinstance(names, str):
            return self._nodes.get(names)
        return {n: self._nodes.get(n) for n in names}

    @property
    def nodes(self) -> dict[str, Node]:
        return dict(self._nodes)

    @property
    def feature_names(self) -> list[str]:
        return list(self._nodes)

    def __repr__(self) -> str:
        return f"<NodeMap {self.title} ({len(self._nodes)} nodes)>"
