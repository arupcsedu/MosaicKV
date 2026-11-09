"""CPU-affined wall/thread diagnostics; never filter observations or replace gates."""

import os
import resource
import threading
from contextlib import contextmanager
from time import perf_counter_ns, thread_time_ns


@contextmanager
def pinned_thread():
    original = set(os.sched_getaffinity(0))
    if not original:
        raise RuntimeError("nonempty Slurm-allowed CPU affinity required")
    chosen = min(original)
    os.sched_setaffinity(0, {chosen})
    try:
        if set(os.sched_getaffinity(0)) != {chosen}:
            raise RuntimeError("main-thread affinity was not applied")
        yield dict(
            original_allowed_cpus=sorted(original),
            pinned_cpu=chosen,
            process_id=os.getpid(),
            thread_id=threading.get_native_id(),
            slurm_cpu_bind_type=os.environ.get("SLURM_CPU_BIND_TYPE", "not_used"),
            slurm_cpu_bind_list=os.environ.get("SLURM_CPU_BIND_LIST", "not_used"),
        )
    finally:
        os.sched_setaffinity(0, original)


def enqueue(
    invoke, count, *, wall=perf_counter_ns, thread=thread_time_ns, usage=resource.getrusage
):
    if type(count) is not int or count < 1:
        raise ValueError("positive fixed launch count required")
    a = usage(resource.RUSAGE_THREAD)
    t0, w0 = thread(), wall()
    for _ in range(count):
        invoke()
    w1, t1 = wall(), thread()
    b = usage(resource.RUSAGE_THREAD)
    elapsed, cpu = (w1 - w0) * 1e-9, (t1 - t0) * 1e-9
    return dict(
        launches=count,
        wall_start_ns=w0,
        wall_end_ns=w1,
        thread_start_ns=t0,
        thread_end_ns=t1,
        host_enqueue_seconds_per_launch=elapsed / count,
        thread_CPU_seconds_per_launch=cpu / count,
        wall_minus_thread_seconds_per_launch=(elapsed - cpu) / count,
        wall_minus_thread_is_NOT_proof_of_scheduler_preemption=True,
        negative_clock_bracket_differences_NOT_clipped=True,
        voluntary_context_switches=b.ru_nvcsw - a.ru_nvcsw,
        involuntary_context_switches=b.ru_nivcsw - a.ru_nivcsw,
        minor_page_faults=b.ru_minflt - a.ru_minflt,
        major_page_faults=b.ru_majflt - a.ru_majflt,
    )
