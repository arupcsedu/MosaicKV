"""Layer-local execution of the FROZEN direct-source precision variant.

Not a new selection policy. The GPU canary must prove payload and membership
identity against the already validated nonstreaming constructor before decoding.
"""

import torch


class StreamingDirectSource:
    def __init__(
        self, adapter, prepared, cache, profile, *, minimum_only=False, construction_geometry=None
    ):
        from mosaickv.evaluation.text_source_storage import TextSourceCapture
        from mosaickv.selection.profile_lifetime import allocate_profile_lifetime

        if type(minimum_only) is not bool:
            raise ValueError("explicit lossless-minimum storage flag required")
        self.minimum_only = minimum_only
        from mosaickv.evaluation.source_row_geometry import GEOMETRIES

        if construction_geometry is not None and (
            not isinstance(construction_geometry, (list, tuple))
            or len(construction_geometry) != 2
            or any(type(x) is not int for x in construction_geometry)
            or tuple(construction_geometry) not in GEOMETRIES
        ):
            raise ValueError("explicit preregistered construction geometry required")
        self.construction_geometry = (
            None if construction_geometry is None else tuple(construction_geometry)
        )

        self.positions = TextSourceCapture(adapter, prepared).positions
        self.length, self.cache = prepared.logical_sequence_length, cache
        self.phase_identity, self.phases, self.identity, self.ids = None, None, None, None
        self.text_phases, self.memberships, self.finished = None, [], False
        if self.length % 2:
            raise NotImplementedError("only audited even-length half-capacity prompts supported")
        if cache is None or len(profile) != len(cache.layers):
            raise ValueError("complete immutable global profile required")
        count = len(profile)
        cost = 32 * 128 * 2 * 2
        allocated = allocate_profile_lifetime(
            tuple(profile),
            (self.length,) * count,
            (1,) * count,
            (cost,) * count,
            cap_bytes=int(count * self.length * cost * 0.5),
            generated_capacity=0,
        )
        lower, extra = divmod(sum(allocated.prompt_positions), count)
        self.capacities = tuple(lower + int(i < extra) for i in range(count))

    def __call__(self, index, attention, original, args, kwargs, output):
        from mosaickv.evaluation.request_precision import REQUEST_POLICY, ranked_positions
        from mosaickv.evaluation.shared_source_format import construct, fit_transport
        from mosaickv.evaluation.source_retile import retile
        from mosaickv.evaluation.text_source_storage import optimize

        x = kwargs.get("hidden_states", args[0] if args else None)
        phase = kwargs.get("position_embeddings")
        if self.finished or index != len(self.memberships) or x is None or phase is None:
            raise ValueError("one ordered original prefill layer required")
        n, h = self.length, original[0].shape[1]
        if x.shape != (1, n, h * 128) or h != 32 or x.dtype != torch.bfloat16:
            raise NotImplementedError("pinned batch-one full-MHA LLaVA source geometry only")
        x = x.detach().contiguous()
        probabilities = output[1].detach()
        if probabilities.shape != (1, h, n, n) or probabilities.dtype != torch.bfloat16:
            raise ValueError("complete original eager prompt probabilities required")
        token = tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride())) for t in phase)
        if self.phases is None:
            self.phase_identity = token
            self.phases = tuple(t.detach().clone().contiguous() for t in phase)
            self.ids = torch.tensor(self.positions, device=x.device, dtype=torch.long)
            self.cache.extra_tensors = (self.ids,)
            self.text_phases = tuple(t.index_select(1, self.ids).contiguous() for t in self.phases)
            self.identity = (
                torch.arange(n, device=x.device, dtype=torch.int32),
                torch.arange(n, device=x.device, dtype=torch.int16)[None].expand(h, n),
            )
        elif token != self.phase_identity:
            raise RuntimeError("streaming source changed the shared original phases")
        window = probabilities[0, :, -REQUEST_POLICY["prompt_window"] :, :].float().mean(-2)
        prior = probabilities.mean(1, keepdim=True).float().mean(-2)[0]
        weight = REQUEST_POLICY["shared_prior_weight"]
        score = (1 - weight) * window + weight * prior
        positions = ranked_positions(score, self.capacities[index], (n - 1,))
        packet = construct(
            x,
            original,
            positions,
            self.positions,
            self.phases,
            self.identity,
            exclude_global_exact=True,
        )
        packet, _ = fit_transport(packet, attention)
        if self.construction_geometry is not None:
            from mosaickv.evaluation.source_row_geometry import geometry

            packet = geometry(packet, *self.construction_geometry)
        text = dict(
            normalized=x.index_select(1, self.ids).detach().contiguous(),
            cos=self.text_phases[0],
            sin=self.text_phases[1],
        )
        packet, _ = optimize(packet, text, (tuple(range(n)),) * h, self.positions, attention)
        if self.construction_geometry is None:
            packet = retile(packet, 4)
        if self.minimum_only:
            from mosaickv.evaluation.exact_minimum_metadata import compact_format

            packet = compact_format(packet)
        self.memberships.append(positions)
        return packet

    def finish(self):
        if len(self.memberships) != len(self.cache.layers):
            raise RuntimeError("all prompt layers must be encoded")
        self.finished, self.ids = True, None
        self.cache.extra_tensors = ()
