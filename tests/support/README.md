# `tests/support/`: read this before writing a test

The rules:

1. **Assert through public interfaces and the fakes' recorders**: frames in and out of `process_frame`, the calls the fakes
   record, the events the service fires, `SynthesiaError` fields, `service.state`/`.metrics`. Never a private attribute. Two
   exceptions: `service._config` via `make_service(**overrides)`, and `TaskManager.current_tasks()` for leak detection.
2. **Durations are inputs, never assertions.** Shrink the knob through its config field, constructor argument or module constant,
   then assert order and eventual outcome with `until()`. Shrink a duration only in a test about that duration elapsing, or one
   that orders against it with at least a 10x margin. An upper wall-clock bound is allowed only against an injected slow operation.
3. **Filter frame lists; don't compare exact sequences.** `expected_down_frames=None`; assert with `frames_of` / `count` /
   `assert_absent` / `assert_ordered`. A new frame type must not break a test that is not about it.
4. **Periodic things are "at least one", never a count**: `BotSpeakingFrame`, TRACE lines, retry sleeps.
5. **Log text is not a contract**, except the user-facing messages a developer acts on (the avatar token and URL errors, the
   bot-kind message, the `avatar_ids` `ValueError`s). Elsewhere: level plus one stable keyword, and only when the log *is* the
   behaviour. Secret greps are always allowed.
6. **Equality for fixed contracts, containment for everything else.** `__all__`, request body keys, header key set, `ErrorType`,
   token attributes: equality, because changing one is a deliberate breaking change. `REGISTERED_EVENTS`, `summary()` keys,
   `CODE_TABLE`, config knobs: superset.
7. **Fakes record; they do not decide.** Any logic in a fake mirrors a behaviour of real rtc or Pipecat, stated in a comment
   beside it.
8. **Every test leaves nothing running.** Context managers own every `TaskManager`, worker, aiohttp session and room;
   `assert_no_dangling(tm)` at exit. Unraisable exceptions and un-awaited coroutines are errors (`pyproject.toml`).
9. **No reruns.** A test that flakes twice moves to `@pytest.mark.quarantine` with an issue.

| Module | Helpers |
|---|---|
| `waiting` | `await until(pred, *, timeout=5.0, interval=0.01, what="")`, `await eventually_equal(get, expected, ...)`, `assert_no_dangling(task_manager)` |
| `frames` | `frames_of(seq, *types)`, `first(seq, type)`, `count(seq, type)`, `assert_absent(seq, type, *, where="")`, `assert_ordered(seq, [T1, T2])` (subsequence, by first occurrence) |
| `events` | `EventRecorder(processor)` → `.events`, `.names()`, `.count(name)`, `.payload(name, index=0)`, `await .wait(name, *, timeout=5.0, count=1)` |
| `held_swap` | `HeldSwapSession(release, **fake_session_kwargs)`: a `FakeAvatarSession` whose first swap waits for `release` |
| `secrets` | `API_KEY`, `LK_API_SECRET`, `LK_TOKEN_MARKER`, `assert_no_secrets(*texts)` |
| `livekit_dev` | *(integration tier only; imports the LiveKit SDK)* `LiveKitDev`, `unique_room_name(request)`, `mint_dev_token(dev, *, identity, room, kind="agent", …)`, `async with connect_room(dev, token) as room:`, `delete_room(dev, name)`, `list_participants(dev, name)`, `list_rooms(dev)` |

For the integration tier there are three more, next to the tests that use them: the `livekit_server` / `room_name` / `launcher`
fixtures (`tests/integration/conftest.py`), `FakeAvatarWorker` (`tests/integration/fake_worker.py`, the oracle: assert on its
recorders, never on livekit-server log text) and the join-mode builders in `tests/integration/_join_modes.py`.
`await launcher.api(**fake_worker_options)` is how a test gets a scripted Synthesia API whose 201 launches a fake worker; put
`launcher.debug_dump()` in every assertion message.

**If you are about to write `await asyncio.sleep(...)` in a test to wait for something, use `until()` instead.** `h.sleep()` is for
*modelling* a delay (letting a hysteresis window elapse with a 10x margin), never for synchronising with a background task.
