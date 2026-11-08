"""Bounded, label-blind optional graph scope; mandatory source nodes survive."""

import numpy as np


def shortlist(attention, required, optional_limit):
    values = np.asarray(attention)
    mandatory = tuple(required)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or type(optional_limit) is not int
        or optional_limit not in (64, 128)
        or len(set(mandatory)) != len(mandatory)
        or any(type(i) is not int or not 0 <= i < len(values) for i in mandatory)
    ):
        raise ValueError(
            "finite prompt attention, unique mandatory IDs and registered limit required"
        )
    ranked = np.argsort(-values, kind="stable")
    excluded = set(mandatory)
    optional = [int(i) for i in ranked if int(i) not in excluded][:optional_limit]
    return tuple(sorted((*mandatory, *optional)))
