# First-step entropy repair without a disposable compressed-model probe

The fixed-plan horizon audit found no consistently useful later-answer-prefix
gain on both consumed inputs. Do not expand that proposer. This change is
runtime-only: preserve the simpler individual-query, residual-prototype policy
and its ORIGINAL entropy threshold, selected rows, weighted prototype values,
source representation, CPU residuals, logical positions and all 128 outputs.

Observe the first actual decode update at the final LLaVA layer. Keep a private
strided copy of its current K until V has been consumed, and preserve the
epilogue input residual before the replay overwrites that buffer. Compute the
same next-token entropy from the actual final hidden state. If no repair is
needed, return the original hidden state unchanged. If it triggers, promote
the actual pinned CPU members, recompute attention and the same real frozen
last-layer suffix once, then return the repaired hidden state to the ordinary
final norm and vocabulary head. No second prefill or disposable whole-model
compressed probe is performed. Last-layer K/V for the current generated token
does not depend on that layer's attention output; no cache update is repeated.

Charge the full private K backing allocation, including stride padding, to the
live KV counter. Record the temporary residual activation separately under the
same activation exclusion as other methods. All private observation owners
must retire immediately. The final and peak counters must still be measured;
this does not certify whole-request GPU peak.

Require bitwise full-128-logit parity with both the original natural-entropy
policy and its forced-restoration control. Forced restoration remains a
mechanics control, never a natural repair or recovered task score. Preserve
the no-repair arm, 32-token parent guard, source retirement and 6,713 auxiliary
GPU bytes. Only BF16 eager, final-layer overlay and the verified uniform reader
are supported. Other policies must fail explicitly, not silently fall back.

Run only the same consumed inputs. Timing is initially a diagnostic single
call; require separate paired repetitions before making any speed claim.
This changes neither the negative quality results nor the blocked status of
fresh evaluation and larger campaigns.
