# Live final-layer parent guard (new safety variant)

The sixteen-image consumed panel completed 192/192 records, but the full overlay
lost 0.595 CIDEr against its storage parent and did not satisfy every parent
stability check. Preserve the negative outcome; do not choose a proposal using
its answers or failure identities. The following change follows a structural
invariant, not fitting those labels.

Only one head of the final decoder layer is modified. At each real update,
capture that head's parent K and V before applying the overlay. Temporarily
restore it in the existing lease to compute the complete native-shape parent
attention distribution/head output; then restore the active lease. Evaluate the
actual frozen final-layer suffix. No second prefill, full-model speculative
decode, benchmark labels, saved reference artifacts, or FullKV offload is used.

Keep the unchanged KL <= .02, max logit deviation <= 1.0 and same-greedy-token
gate. If a correction fails, emit the live **storage-parent** counterfactual for
that step. This is an explicit safety veto, not FullKV fallback or fabricated
residual repair. Existing first-step pinned residual repair remains separately
identified. Every earlier layer, generated K/V and subsequent input remains the
parent's by induction IF counterfactual arithmetic and emission controls pass.

Require actual pretrained parity of every parent counterfactual against an
untouched parent run, all 128 emitted tokens, same-prefix guards, request
isolation and measured backing storage. Do not assert the induction merely
because the code executes. Charge the routed-head K/V copies to the cache
counter; record attention/logit/hidden workspace separately. Diagnostic CPU
logit exports are activation artifacts, not residual KV or free latency.

Initially test only the same two heavily consumed inputs. Report veto frequency
and useful accepted corrections, not just parity. Do not expand or claim all-six
superiority if the guard simply disables the method or adds unacceptable cost.
No improvement in task score or net source retirement is implied by this guard.
