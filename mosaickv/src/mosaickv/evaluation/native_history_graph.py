"""Private immutable-history graphs, not captured attention or model decoding."""

import torch


class Entries:
    """CPU-only bounded ledger; retain owners until explicit epoch retirement."""

    def __init__(self, layers):
        if type(layers) is not int or not 1 <= layers <= 32:
            raise ValueError("bounded audited layer count required")
        self.maximum, self.rows, self.closed = 2 * layers, {}, False
        self.captures = self.replays = self.retired = self.maximum_live = 0

    def get(self, key, signature):
        if self.closed:
            raise RuntimeError("retired history graph table")
        value = self.rows.get(key)
        if value is not None and value[0] != signature:
            raise RuntimeError("immutable history graph bindings changed")
        return None if value is None else value[1]

    def add(self, key, signature, graph, packet, attention):
        if self.closed or key in self.rows or len(self.rows) >= self.maximum:
            raise RuntimeError("bounded private history graph table exhausted or duplicate")
        self.rows[key] = (signature, graph, packet, attention)
        self.captures += 1
        self.maximum_live = max(self.maximum_live, len(self.rows))

    def discard(self, packet_ids):
        keys = [key for key in self.rows if key[0] in packet_ids]
        for key in keys:
            value = self.rows.pop(key)
            value[1].reset()
            self.retired += 1
        return len(keys)

    def close(self):
        self.discard({key[0] for key in self.rows})
        self.closed = True


class Reader:
    def __init__(self, state):
        self.state, self.table = state, Entries(len(state.pool.cache.layers))
        self.direct_calls = self.warm_calls = self.additional_tensor_bytes = 0

    def eligible(self, target):
        pool = self.state.pool
        return bool(
            target.device.type == "cuda"
            and pool is not None
            and not pool.closed
            and not pool.history_borrowed
            and pool.buffer is not None
            and target.untyped_storage().data_ptr() == pool.buffer.untyped_storage().data_ptr()
        )

    def project(self, packet, attention, component, target, repairs):
        if not self.eligible(target):
            self.direct_calls += 1
            return self.state.engine.project(packet, attention, component, target, repairs)
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("private eager inference outside any other capture required")
        key = id(packet), component
        signature = (
            id(attention),
            target.data_ptr(),
            tuple(target.shape),
            tuple(target.stride()),
            target.dtype,
            target.device,
        )
        graph = self.table.get(key, signature)
        if graph is None:
            self.state.engine.project(packet, attention, component, target, repairs)
            self.warm_calls += 1
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            current, stream = torch.cuda.current_stream(), torch.cuda.Stream()
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        self.state.engine.project(packet, attention, component, target, repairs)
                    finally:
                        graph.capture_end()
                current.wait_stream(stream)
                torch.cuda.synchronize()
                extra = torch.cuda.memory_allocated() - before
                if extra:
                    raise RuntimeError(f"history graph introduced {extra} unowned tensor bytes")
                self.table.add(key, signature, graph, packet, attention)
            except BaseException:
                graph.reset()
                raise
        graph.replay()
        self.table.replays += 1

    def discard(self, packets):
        ids = {id(packet) for packet in packets}
        if not any(key[0] in ids for key in self.table.rows):
            return
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        self.table.discard(ids)
        torch.cuda.synchronize()
        if torch.cuda.memory_allocated() != before:
            raise RuntimeError("history graph epoch retirement changed owned tensor bytes")

    def close(self):
        if self.table.closed:
            return
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        self.table.close()
        torch.cuda.synchronize()
        if torch.cuda.memory_allocated() != before:
            raise RuntimeError("history graph retirement changed existing prompt owners")

    def record(self):
        return dict(
            enabled=True,
            captures=self.table.captures,
            replays=self.table.replays,
            retired_graphs=self.table.retired,
            maximum_live_graphs=self.table.maximum_live,
            live_graphs=len(self.table.rows),
            retired=self.table.closed,
            direct_construction_calls=self.direct_calls,
            warm_calls=self.warm_calls,
            additional_GPU_tensor_bytes=self.additional_tensor_bytes,
            attention_and_model_layers_not_captured=True,
            immutable_history_only=True,
            capture_and_retirement_in_request_time=True,
        )
