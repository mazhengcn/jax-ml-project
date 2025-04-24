"""Splits related API."""

import dataclasses
import math
import re

# <split_name>[<split_selector>] (e.g. `train[54%:]`)
_SUB_SPEC_RE = re.compile(
    r"""^
    (?P<split_name>[\w-]+)
    (\[
      (?P<split_selector>[\d\w%:.-]+)
    \])?
    $""",
    re.VERBOSE,  # Ignore whitespace
)
# <val><unit> (e.g. `-54%`)
_SLICE_RE = re.compile(
    r"""^
    (
        (?P<val>-?[\d_.]+)
        (?P<unit>(?:%|shard))?
    )?
    $""",
    re.VERBOSE,  # Ignore whitespace
)

_ADDITION_SEP_RE = re.compile(r"\s*\+\s*")

PERCENT_BOUNDARY = 100


@dataclasses.dataclass(frozen=True)
class AbsoluteInstruction:
    """A machine friendly slice: defined absolute positive boundaries."""

    splitname: str
    from_: int  # uint (starting index).
    to: int  # uint (ending index).


@dataclasses.dataclass(frozen=True)
class ReadInstruction:
    """Represents a read instruction for a dataset split."""

    split_name: str
    from_: int | float | None = None
    to: int | float | None = None
    unit: str = "abs"
    rounding: str = "closest"

    def __post_init__(self) -> None:
        """Post-initialization to validate the ReadInstruction instance."""
        # Perform validation
        allowed_units = ["%", "abs", "shard"]
        allowed_rounding = ["closest", "pct1_dropremainder"]
        if self.unit not in allowed_units:
            error_message = f"Unit should be one of {allowed_units}. Got {self.unit!r}"
            raise ValueError(error_message)
        if self.rounding not in allowed_rounding:
            error_message = (
                f"Rounding should be one of {allowed_rounding}. Got: {self.rounding!r}"
            )
            raise ValueError(error_message)
        if self.unit == "%" and (
            abs(self.from_ or 0) > PERCENT_BOUNDARY
            or abs(self.to or 0) > PERCENT_BOUNDARY
        ):
            error_message = (
                "When unit=%, percent slice boundaries should be "
                f"in [-{PERCENT_BOUNDARY}, {PERCENT_BOUNDARY}]. Got: {self}"
            )
            raise ValueError(error_message)

    def __repr__(self) -> str:
        """Return a string representation of the ReadInstruction instance."""
        unit = "" if self.unit == "abs" else self.unit
        from_ = "" if self.from_ is None else f"{self.from_:g}{unit}"
        to = "" if self.to is None else f"{self.to:g}{unit}"
        slice_str = "" if self.from_ is None and self.to is None else f"[{from_}:{to}]"
        rounding = f", rounding={self.rounding!r}" if self.unit == "%" else ""
        return f"ReadInstruction('{self.split_name}{slice_str}'{rounding})"


def get_split_instruction(spec: str, num_examples: int) -> AbsoluteInstruction:
    """Parse a split string into an AbsoluteInstruction."""
    rel_instr = _str_to_relative_instruction(spec)
    return _rel_to_abs_instr(rel_instr, num_examples)


def _str_to_relative_instruction(spec: str) -> ReadInstruction:  # noqa: C901
    """Return ReadInstruction for given string."""
    # <split_name>[<split_selector>] (e.g. `train[54%:]`)
    res = _SUB_SPEC_RE.match(spec)
    err_msg = (
        f"Unrecognized split format: {spec!r}. See format at "
        "https://www.tensorflow.org/datasets/splits"
    )
    if not res:
        raise ValueError(err_msg)
    split_name = res.group("split_name")
    split_selector = res.group("split_selector")

    if split_name == "all":
        if split_selector:
            error_message = (
                f"{split_name!r} does not support slice. Please open a github issue "
                "if you need this feature."
            )
            raise NotImplementedError(error_message)

        split_selector = None

    if split_selector is None:  # split='train'
        from_ = None
        to = None
        unit = "abs"
    else:  # split='train[x:y]' or split='train[x]'
        slices = [_SLICE_RE.match(x) for x in split_selector.split(":")]
        # Make sure all slices are valid, and at least one is not empty
        if not all(slices) or not any(
            x.group(0) for x in slices if x is not None
        ):  # re-none
            raise ValueError(err_msg)
        if len(slices) == 1:  # split='train[x]'
            (from_match,) = slices
            from_ = from_match["val"]  # type: ignore[attr-defined]
            to = int(from_) + 1
            unit = from_match["unit"] or "abs"  # type: ignore[attr-defined]
            if unit != "shard":
                error_message = "Absolute or percent only support slice syntax."
                raise ValueError(error_message)
        elif len(slices) == 2:  # split='train[x:y]'  # noqa: PLR2004
            from_match, to_match = slices
            from_ = from_match["val"]  # type: ignore[attr-defined]
            to = to_match["val"]  # type: ignore[attr-defined]
            unit = from_match["unit"] or to_match["unit"] or "abs"  # type: ignore[attr-defined]
        else:
            raise ValueError(err_msg)

    if from_ is not None:
        from_ = float(from_) if unit == "%" else int(from_)
    if to is not None:
        to = float(to) if unit == "%" else int(to)

    return ReadInstruction(
        split_name=split_name, rounding="closest", from_=from_, to=to, unit=unit
    )


def _pct_to_abs_pct1(boundary: float, num_examples: int) -> float:
    # Using math.trunc here, since -99.5% should give -99%, not -100%.
    if num_examples < 100:  # noqa: PLR2004
        msg = (
            'Using "pct1_dropremainder" rounding on a split with less than 100 '
            "elements is forbidden: it always results in an empty dataset."
        )
        raise ValueError(msg)
    return boundary * math.trunc(num_examples / 100.0)


def _pct_to_abs_closest(boundary: float, num_examples: int) -> int:
    return int(round(boundary * num_examples / 100.0))


def _rel_to_abs_instr(
    rel_instr: ReadInstruction, num_examples: int
) -> AbsoluteInstruction:
    """Return _AbsoluteInstruction instance for given RelativeInstruction.

    Args:
      rel_instr: ReadInstruction instance.
      num_examples: int, number of examples in the dataset.

    """
    pct_to_abs = (
        _pct_to_abs_closest if rel_instr.rounding == "closest" else _pct_to_abs_pct1
    )
    split = rel_instr.split_name
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == "%":
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    elif rel_instr.unit == "abs":
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    else:
        error_message = f"Invalid split unit: {rel_instr.unit}"
        raise ValueError(error_message)
    if abs(from_) > num_examples or abs(to) > num_examples:
        msg = "Requested slice [{}:{}] incompatible with {} examples.".format(
            from_ or "",
            to or "",
            num_examples,
        )
        raise ValueError(msg)
    if from_ < 0:
        from_ = num_examples + from_
    elif from_ == 0:
        from_ = None
    if to < 0:
        to = num_examples + to
    elif to == num_examples:
        to = None
    return AbsoluteInstruction(split, from_, to)  # type: ignore[arg-type]
