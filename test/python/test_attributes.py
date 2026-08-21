# RUN: python %s

# Copyright 2026 The Torch-Spyre Authors.
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

"""Builder tests for ktdp attributes.

These construct attributes through the bindings rather than by parsing text,
so a rename of a mnemonic or an enum keyword breaks at the API level instead
of silently drifting away from a hand-written `#ktdp.…` literal.

`get` is verified and requires a Location (as the upstream MLIR type builders
do), so these run inside one and the invalid cases assert on the diagnostic.
"""

from typing import TYPE_CHECKING

from mlir_ktdp.ir import Attribute, Location
from mlir_ktdp.tools import ktdp_context
import mlir_ktdp.dialects.ktdp as ktdp

# MLIRError is created at runtime by _mlir_libs/__init__.py:_site_initialize
# and assigned onto mlir_ktdp.ir, so it appears in no stub and no static
# analyzer can see it. Alias Exception for type checking only; at runtime this
# always binds the real class.
if TYPE_CHECKING:
    MLIRError = Exception
else:
    from mlir_ktdp.ir import MLIRError

MemorySpaceKind = ktdp.MemorySpaceKind


# ---------------------------------------------------------------------------
# MemorySpaceAttr: (kind, ct_id) -> expected assembly
# ---------------------------------------------------------------------------

MEMORY_SPACE_CASES = [
    ((MemorySpaceKind.global_,  None), "#ktdp.memory_space<global>"),
    ((MemorySpaceKind.ct_local, None), "#ktdp.memory_space<ct_local>"),
    ((MemorySpaceKind.ct_local, 0),    "#ktdp.memory_space<ct_local, ct_id = 0>"),
    ((MemorySpaceKind.ct_local, 7),    "#ktdp.memory_space<ct_local, ct_id = 7>"),
    # -1 is the "unspecified" sentinel, so it round-trips to no ct_id.
    ((MemorySpaceKind.ct_local, -1),   "#ktdp.memory_space<ct_local>"),
]

# (kind, ct_id) combinations the attribute verifier must reject.
MEMORY_SPACE_INVALID = [
    (MemorySpaceKind.global_,  0),   # ct_id is only valid for ct_local
    (MemorySpaceKind.global_,  7),
    (MemorySpaceKind.ct_local, -5),  # ct_id must be non-negative
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

with ktdp_context(), Location.unknown():
    for (kind, ct_id), expected in MEMORY_SPACE_CASES:
        attr = ktdp.MemorySpaceAttr.get(kind, ct_id=ct_id)
        assert str(attr) == expected, f"\nactual:   {attr}\nexpected: {expected}"

        # Builder output must match what the parser produces for the same text.
        assert attr == Attribute.parse(expected), f"{attr} != parse({expected})"

        # Accessors mirror the parameters. -1 reads back as unspecified.
        assert attr.kind == kind, f"{attr.kind} != {kind}"
        expected_ct_id = ct_id if (ct_id is not None and ct_id >= 0) else None
        assert attr.ct_id == expected_ct_id, f"{attr.ct_id} != {expected_ct_id}"

        # kind is the tablegen-generated IntEnum, not a string or a bare int.
        assert isinstance(attr.kind, MemorySpaceKind), type(attr.kind)

    # ct_id defaults to unspecified when omitted entirely.
    assert str(ktdp.MemorySpaceAttr.get(MemorySpaceKind.global_)) == \
        "#ktdp.memory_space<global>"

    # A parsed attribute downcasts to the concrete class via its typeid.
    parsed = Attribute.parse("#ktdp.memory_space<ct_local, ct_id = 2>")
    assert isinstance(parsed, ktdp.MemorySpaceAttr), type(parsed)
    assert parsed.kind == MemorySpaceKind.ct_local
    assert parsed.ct_id == 2

    # Verifier failures surface as MLIRError, not a crash or a null attribute.
    for kind, ct_id in MEMORY_SPACE_INVALID:
        try:
            attr = ktdp.MemorySpaceAttr.get(kind, ct_id=ct_id)
        except MLIRError:
            pass
        else:
            raise AssertionError(
                f"expected MLIRError for kind={kind}, ct_id={ct_id}, got {attr}")

    # An explicit loc= overrides the ambient location, and the diagnostic is
    # emitted there -- so a bad memory space points at the code that built it
    # rather than reporting "unknown".
    loc = Location.file("emitter.py", 42, 5)
    assert str(ktdp.MemorySpaceAttr.get(MemorySpaceKind.global_, loc=loc)) == \
        "#ktdp.memory_space<global>"
    try:
        ktdp.MemorySpaceAttr.get(MemorySpaceKind.global_, ct_id=3, loc=loc)
    except MLIRError as e:
        assert "emitter.py" in str(e), str(e)
    else:
        raise AssertionError("expected MLIRError for global with ct_id")

    # get_unchecked skips verification, so it takes a context and no Location.
    for (kind, ct_id), expected in MEMORY_SPACE_CASES:
        kwargs = {} if ct_id is None else {"ct_id": ct_id}
        unchecked = ktdp.MemorySpaceAttr.get_unchecked(kind, **kwargs)
        assert str(unchecked) == expected, f"{unchecked} != {expected}"
        assert unchecked == ktdp.MemorySpaceAttr.get(kind, **kwargs), \
            "get and get_unchecked must agree on valid input"
