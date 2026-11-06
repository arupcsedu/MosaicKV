# Same-value verified constructor handoff (consumed-input variant)

The existing construction profile shows repeated source projection for
transport verification, cached text, and native-format reference construction.
This is a default-off execution transaction, not a new selector. Test two
separate ablations: `same_buffer`, then `handoff`. Keep the frozen source
precision variant, modes, selection, all mandatory positions and numerical
thresholds. All six main algorithm mechanisms remain inactive in this variant.

`same_buffer` applies the original sparse correction to the already projected
component instead of projecting it a second time. Keep the original complete
closeness predicate. The audited path also performs the original second
projection and requires bitwise equality; it does not weaken that check.

`handoff` consumes that verified component while it is live. Extract only
compact text rows and calculate counted native execution corrections before
releasing it. Never retain a full reference across components or layers. Use
the same exact protected rows, deterministic lossless text coding and original
layout. Keep the preconversion frozen-packet digest check. The audit compares
the final OUTER packet, including cached text, against the original decoder at
tail lengths 0/1/127, every layer and both K/V components. This is essential:
constructor and reader head groups differ, and text stripping changes the
base decoder where the cached overlay later overwrites it. A mismatch rejects
the variant; no tolerance is changed. Require all 128 parent logits bitwise.

Do not assume different head groups produce identical words: after the
original transport closeness check, replay the exact registered parent
geometry into that same component owner before handing it to text/native
construction. Count these 64 replays. This retains the parent arithmetic and
still avoids separate full reconstruction for cached text and native fitting.

Charge constructor-scope setup and retirement in `total_wall_seconds` for
every arm, including ineligible baselines and the original constructor. Keep
the original core clock and measured scope overhead separately. Do not reuse
historical timing as a paired control under this explicitly wider boundary.

The handoff replaces asynchronous native fitting with an inline bounded
transaction; record that policy change and its effect. Private engine, compact
text, correction storage, CPU index metadata and whole-request Torch peaks
must be reported. The unchanged correction caps are 32 KiB/layer and
1 MiB/request. Initial/final/decode-owned backing storage must be measured,
not inferred from views. Whole-request vendor/GPU peak remains uncertified.

Only the two consumed 600/604-token, full-MHA LLaVA inputs are supported by
this diagnostic. Baselines without a shared-source constructor remain
truthful, allocation-free no-ops; common history/normalization optimizations
remain offered identically. No fresh data, quality sweep or paper superiority
claim is authorized by a constructor or timing diagnostic.
