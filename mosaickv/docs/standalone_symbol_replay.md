# Standalone source-symbol replay correction

The sixteen-consumed-image full-head check completed three and six cases before
two ordinary-parent requests failed in common source-code conversion. They did
not fail the full-head numerical comparison. The broad packet inventory exposed
an old assumption: every non-raw compressed packet required a previous-layer
stream. `DeviceSymbols(has_previous=False)` is compressed but self-contained.
Its legitimate first occurrence therefore raised `KeyError: 'source'`.

The conversion verification and both prepared/captured replay now resolve the
packet's explicit dependency. Legacy compressed `DeviceCodes` continue to require
a predecessor; missing required streams and wrong shapes remain hard errors.
No dummy source, uncompressed substitute, weakened check or changed serialized
payload is introduced. Independent packets decode with `None` as intended by
their existing native unpack kernel. Scratch, offsets, selector bytes and source
retirement remain charged. The correction applies equally to storage-only and
the six-stage hybrid; it is not an adaptive quality improvement.

CPU regressions cover self-contained first-layer compressed packets, ordinary
and prepared replay, captured transaction descriptors, three complete epochs,
disjoint request scratch, exact backing accounting and missing-required-source
rejection. Retry the same consumed correctness panel after the clean fix, not
a new quality cohort. Only a complete numerical/veto/isolation pass may release
the conditional timing study. Negative earlier roots remain immutable.
