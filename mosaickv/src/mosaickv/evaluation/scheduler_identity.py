"""Current execution identity, never a copied parent scheduler allocation."""

import os


def scheduler_identity(parent, environ=None):
    env = os.environ if environ is None else environ
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
    values = {k: env.get(k) for k in keys}
    if any(
        v is not None and (not isinstance(v, str) or not v.isdecimal()) for v in values.values()
    ):
        raise ValueError("decimal current Slurm identities required")
    job, array, task = (values[k] for k in keys)
    if (array is None) != (task is None) or (array is not None and job is None):
        raise ValueError("complete current array/job identity required")
    old = {
        k: parent[k]
        for k in (
            "slurm_job_id",
            "slurm_array_job_id",
            "slurm_array_task_id",
            "slurm_array_element",
        )
        if k in parent
    }
    return dict(
        slurm_job_id=job or "not_slurm",
        slurm_array_job_id=array or "not_used",
        slurm_array_task_id=task if task is not None else "not_used",
        slurm_array_element=f"{array}_{task}" if array is not None else "not_used",
        parent_scheduler_identity=old,
        scheduler_identity_source="current_process_environment",
    )
