#include <pybind11/pybind11.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace py = pybind11;
namespace ca = c10::cuda::CUDACachingAllocator;

namespace {
constexpr std::uint64_t MAX_CAPACITY = 2000000;
enum Phase { UNUSED = 0, STARTING = 1, RUNNING = 2, STOPPING = 3, STOPPED = 4, FAILED = 5 };

struct NumericEvent {
  std::uint64_t sequence;
  std::int64_t device;
  std::uint64_t address;
  std::uint64_t size;
  std::uint64_t stream;
  std::uint64_t pool_first;
  std::uint64_t pool_second;
  std::uint64_t approximate_time;
  std::int64_t action;
};
static_assert(std::is_trivially_copyable<NumericEvent>::value, "numeric callback storage only");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "lock-free counters required");
static_assert(std::atomic<bool>::is_always_lock_free, "lock-free enabled flag required");
static_assert(std::atomic<int>::is_always_lock_free, "lock-free phase required");

struct State {
  std::vector<NumericEvent> slots;  // Resized exactly once BEFORE attachment.
  std::atomic<std::uint64_t> next{0};
  std::atomic<std::uint64_t> inflight{0};
  std::atomic<bool> enabled{false};
  std::atomic<bool> overflow{false};
  std::atomic<int> phase{UNUSED};
  ca::CUDAAllocator* attached_allocator = nullptr;
  bool attached = false;
};

State& global_state() {
  // Deliberately immortal. The allocator has no detach API. Neither this
  // state nor its buffer is ever freed/reused while callbacks can exist.
  static State* const instance = new State();
  return *instance;
}

// CALLBACK BEGIN: under allocator lock; no GIL/Python/CUDA/locks/heap operations.
void record(State* const state, const ca::TraceEntry& entry) noexcept {
  if (!state->enabled.load()) return;
  state->inflight.fetch_add(1);
  if (!state->enabled.load()) {
    state->inflight.fetch_sub(1);
    return;
  }
  const auto ticket = state->next.fetch_add(1);
  if (ticket < state->slots.size()) {
    state->slots[ticket] = NumericEvent{
        ticket,
        static_cast<std::int64_t>(entry.device_),
        static_cast<std::uint64_t>(entry.addr_),
        static_cast<std::uint64_t>(entry.size_),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(entry.stream_)),
        static_cast<std::uint64_t>(entry.mempool_.first),
        static_cast<std::uint64_t>(entry.mempool_.second),
        static_cast<std::uint64_t>(entry.time_.approx_t_),
        static_cast<std::int64_t>(entry.action_)};
  } else {
    state->overflow.store(true);
  }
  state->inflight.fetch_sub(1);
}
// CALLBACK END

bool quiesce(State& state) {
  // All atomics intentionally sequentially consistent: if the callback's
  // second enabled check sees true, stop must observe its inflight entry.
  // A delayed first-check callback can only see false at its second check;
  // immutable process-lifetime storage makes that late return safe.
  state.enabled.store(false);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (state.inflight.load() != 0) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::yield();  // Outside allocator lock and outside callback.
  }
  return true;
}

const char* action_name(std::int64_t action) {
  switch (static_cast<ca::TraceEntry::Action>(action)) {
    case ca::TraceEntry::ALLOC: return "alloc";
    case ca::TraceEntry::FREE_REQUESTED: return "free_requested";
    case ca::TraceEntry::FREE_COMPLETED: return "free_completed";
    case ca::TraceEntry::SEGMENT_ALLOC: return "segment_alloc";
    case ca::TraceEntry::SEGMENT_FREE: return "segment_free";
    case ca::TraceEntry::SEGMENT_MAP: return "segment_map";
    case ca::TraceEntry::SEGMENT_UNMAP: return "segment_unmap";
    case ca::TraceEntry::SNAPSHOT: return "snapshot";
    case ca::TraceEntry::OOM: return "oom";
  }
  return "unknown";
}

py::dict report(const State& state, bool drained, bool backend_unchanged, bool synthetic) {
  const auto attempted = state.next.load();
  const auto count = std::min<std::uint64_t>(attempted, state.slots.size());
  py::list events;
  bool unknown_action = false;
  if (drained) {
    for (std::uint64_t i = 0; i < count; ++i) {
      const auto& event = state.slots[i];
      py::dict row;
      row["sequence"] = py::int_(event.sequence);
      row["device"] = py::int_(event.device);
      row["addr"] = py::int_(event.address);
      row["size"] = py::int_(event.size);
      row["stream"] = py::int_(event.stream);
      py::list pool;
      pool.append(py::int_(event.pool_first)); pool.append(py::int_(event.pool_second));
      row["pool"] = pool;
      row["mempool"] = pool;  // Explicit schema alias, not a second owner.
      row["approx_time"] = py::int_(event.approximate_time);
      row["action_code"] = py::int_(event.action);
      const auto* name = action_name(event.action);
      if (std::strcmp(name, "unknown") == 0) unknown_action = true;
      row["action"] = name;
      events.append(row);
    }
  }
  py::dict result;
  result["schema"] = "native_numeric_allocator_trace_v1";
  result["events"] = events;
  result["count"] = py::int_(drained ? count : 0);
  result["capacity"] = py::int_(state.slots.size());
  result["attempted_events"] = py::int_(attempted);
  result["overflow"] = state.overflow.load();
  result["writes_quiesced"] = drained;
  result["stop_wait_timeout"] = !drained;
  result["backend_unchanged"] = backend_unchanged;
  result["unknown_action"] = unknown_action;
  result["complete"] = drained && backend_unchanged && !state.overflow.load() && !unknown_action;
  result["synthetic_test_only"] = synthetic;
  result["timestamp_domain"] = "raw_torch_getApproximateTime_not_epoch_or_duration";
  result["sequence_semantics"] = "global_callback_ticket_per_device_order_not_CUDA_execution_order";
  result["size_semantics"] = "alloc_and_free_requested_bytes_segment_physical_bytes_OOM_request";
  result["callback_history_required"] = false;
  result["automatic_KV_roles_assigned"] = false;
  result["whole_request_KV_peak_certified"] = false;
  return result;
}

bool uncached_environment() {
  for (const auto* name : {"PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NO_HIP_MEMORY_CACHING"}) {
    const auto* value = std::getenv(name);
    if (value && std::strcmp(value, "0") != 0) return true;
  }
  return false;
}

py::dict start(std::uint64_t capacity) {
  if (capacity == 0 || capacity > MAX_CAPACITY) throw std::invalid_argument("bounded capacity1..2000000 required");
  auto* allocator = ca::get();
  if (!allocator || allocator->name() != "native" || !allocator->isEnabled() || uncached_environment())
    throw std::runtime_error("enabled native caching allocator and unset/0 uncached flags required");
  if (!allocator->initialized())
    throw std::runtime_error("allocator not initialized; call torch.cuda.init before attachment");
  auto& state = global_state();
  int expected = UNUSED;
  if (!state.phase.compare_exchange_strong(expected, STARTING))
    throw std::runtime_error("one attachment/session per process; restart is forbidden");
  try {
    state.slots.resize(capacity);
    state.attached_allocator = allocator;
    {
      py::gil_scoped_release release;
      State* const stable = &state;
      allocator->attachAllocatorTraceTracker(
          [stable](const ca::TraceEntry& entry) noexcept { record(stable, entry); });
    }
    state.attached = true;
    state.enabled.store(true);
    state.phase.store(RUNNING);
  } catch (...) {
    state.enabled.store(false); state.phase.store(FAILED); throw;
  }
  py::dict result;
  result["capacity"] = py::int_(capacity);
  result["attached"] = true;
  result["one_shot"] = true;
  result["cursor_at_return"] = py::int_(state.next.load());
  result["callback_history_required"] = false;
  result["host_buffer_bytes"] = py::int_(capacity * sizeof(NumericEvent));
  return result;
}

py::dict stop() {
  auto& state = global_state();
  int expected = RUNNING;
  if (!state.phase.compare_exchange_strong(expected, STOPPING))
    throw std::runtime_error("exactly one running recorder required");
  bool drained;
  {
    py::gil_scoped_release release;
    drained = quiesce(state);
  }
  state.phase.store(drained ? STOPPED : FAILED);
  auto* allocator = ca::get();
  const bool unchanged = allocator == state.attached_allocator && allocator->isEnabled() && !uncached_environment();
  return report(state, drained, unchanged, false);
}

#ifdef MOSAICKV_NATIVE_ALLOCATOR_TESTING
py::dict test_only_capture(py::list input, std::uint64_t capacity, int threads, std::int64_t stop_after) {
  if (capacity == 0 || capacity > MAX_CAPACITY || threads < 1 || threads > 16)
    throw std::invalid_argument("bounded synthetic capacity/thread count required");
  std::vector<NumericEvent> originals;
  for (const auto& item : input) {
    const auto row = py::cast<py::dict>(item);
    auto pool = py::cast<py::tuple>(py::tuple(row["pool"]));
    if (pool.size() != 2) throw std::invalid_argument("two integer pool fields required");
    originals.push_back(NumericEvent{0, py::cast<std::int64_t>(row["device"]),
      py::cast<std::uint64_t>(row["addr"]), py::cast<std::uint64_t>(row["size"]),
      py::cast<std::uint64_t>(row["stream"]), py::cast<std::uint64_t>(pool[0]),
      py::cast<std::uint64_t>(pool[1]), py::cast<std::uint64_t>(row["approx_time"]),
      py::cast<std::int64_t>(row["action_code"])});
  }
  State state; state.slots.resize(capacity); state.enabled.store(true);
  bool drained = false;
  {
    py::gil_scoped_release release;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < threads; ++worker) {
      workers.emplace_back([&, worker] {
        for (std::size_t i = worker; i < originals.size(); i += threads) {
          const auto& e = originals[i];
          ca::TraceEntry entry(static_cast<ca::TraceEntry::Action>(e.action),
            static_cast<c10::DeviceIndex>(e.device), e.address, e.size,
            reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(e.stream)),
            {e.pool_first, e.pool_second}, static_cast<c10::approx_time_t>(e.approximate_time));
          record(&state, entry);
        }
      });
    }
    if (stop_after >= 0) {
      while (state.next.load() < static_cast<std::uint64_t>(stop_after) && state.next.load() < originals.size())
        std::this_thread::yield();
      drained = quiesce(state);
    }
    for (auto& worker : workers) worker.join();
    if (stop_after < 0) drained = quiesce(state);
  }
  return report(state, drained, true, true);
}
#endif
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("start", &start);
  module.def("stop", &stop);
  module.def("cursor", [] { return global_state().next.load(); });
  module.def("metadata", [] {
    py::dict result;
    result["schema"] = "native_numeric_allocator_trace_v1";
    result["maximum_capacity"] = py::int_(MAX_CAPACITY);
    result["numeric_event_bytes"] = py::int_(sizeof(NumericEvent));
    result["one_process_lifetime_session"] = true;
    result["detach_supported"] = false;
    result["callback_requires_GIL"] = false;
    result["callback_history_required"] = false;
    result["approx_time_is_epoch"] = false;
    result["automatic_KV_roles_assigned"] = false;
#ifdef MOSAICKV_NATIVE_ALLOCATOR_TESTING
    result["synthetic_test_build"] = true;
#else
    result["synthetic_test_build"] = false;
#endif
    return result;
  });
#ifdef MOSAICKV_NATIVE_ALLOCATOR_TESTING
  module.def("_test_only_capture", &test_only_capture, py::arg("events"), py::arg("capacity"),
             py::arg("threads") = 1, py::arg("stop_after") = -1);
#endif
}
