# Common encoded-source transport across all primary baseline paths

The encoded-source control passed its preregistered exact-output, metadata,
cache-peak and <=1.05 latency-ratio gates on the consumed EviCastKV and Top-K
inputs. This follow-up applies the same eligible transport to Full-position,
Top-K, PrefixKV, VL-Cache, LOOK-M and EviCastKV paths. LOOK-M's transformed K/V
cannot be regenerated from original source; its original lossless K/V codec
is retained, never substituted with unmerged values.

All methods retain the same original source-slot capacities, prompts, media,
precision, attention implementation, output schedule, mandatory policy and
seed as the earlier diagnostic. Realized encoded bytes differ and are reported;
this is not a shared-physical-byte selection-policy comparison. Full-position
rows use a lossless transport and are not ordinary FullKV memory measurements.
Ordinary FullKV remains the untouched quality/parity reference, never a
compressed-method latency competitor.

Each path must preserve original 32-token logits, pass 32/128-token history
controls, and charge checkpoint metadata and all observed reconstruction
scratch. The same bounded four-layer history sealing is offered to every
method. Five warmups and twenty randomized paired blocks measure decode only.
No timing result includes original prefill, selection or constructor; do not
sum different-run medians into an end-to-end claim. Whole-request HBM peak is
not certified. Two consumed examples cannot establish independent quality
preservation or universal superiority. No fresh/large campaign is authorized.
