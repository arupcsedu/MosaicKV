# Paired full-request key-fold screen

The preceding A-B-A check passed all 128 parent vocabulary-logit vectors,
storage isolation, all 4032 folded key projections and the extra 8 KiB charged
workspace on both consumed inputs. This commit changes only controller,
documentation and CPU tests. No production or numerical module is changed.

On those same inputs compare five readers: EviCastKV fused, native, joint-key;
PrefixKV original-graph and optimized-graph. All receive prepared sparse-patch
dispatch and event-sum accounting. Native conversion uses its direct validated
constructor. The new feature is not silently enabled in controls. All arms
receive a fresh source-packet digest and actual initial/final/decode-owned peak
audit. Then run five warmups and three timed repetitions with randomized paired
method order, one isolated A100 per worker, 12 CPUs and 20-minute allocation.
Timing covers complete media processing, image/language prefill, construction,
source copies, capture, decode and retirement. Failed/incomplete rows persist.

This is a screening diagnostic, not formal acceptance. The formal target still
requires adjusted paired upper ratio <=0.96 against every original and optimized
baseline, five warmups and 20 measured repetitions. Independent benchmark N=0,
whole-request GPU peak uncertified. No fresh/calibration/paper campaign is
submitted by this screen. The next engineering action must follow measured
phase costs rather than a broader unregistered optimization sweep.
