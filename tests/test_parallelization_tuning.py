import asyncio
import re
import time

from gabriel.utils import openai_utils


def test_example_prompt_is_plain_text(capsys):
    prompt = "Line one\nLine two"
    openai_utils._display_example_prompt(prompt, verbose=True)
    output = capsys.readouterr().out
    assert "===== Example prompt =====" in output
    assert "Line one" in output and "Line two" in output
    assert "<details" not in output


def test_usage_overview_compact_printout(capsys):
    openai_utils._print_usage_overview(
        prompts=["hello", "world"],
        n=1,
        max_output_tokens=None,
        model="gpt-5.4-mini",
        use_batch=False,
        n_parallels=4,
        estimated_output_tokens_per_prompt=32,
        verbose=True,
        rate_headers={"limit_requests": "20", "limit_tokens": "2000"},
        heading="Usage check",
        show_prompt_stats=False,
    )
    output = capsys.readouterr().out
    assert "Usage check" in output
    assert "Prompts:" not in output
    assert "<summary" not in output


def test_wait_based_cap_dampens_reductions():
    now = time.time()
    cap, last_adjust, changed = openai_utils._smooth_wait_based_cap(
        current_cap=100,
        candidate_cap=40,
        now=now,
        last_adjust=0.0,
        limiter_pressure=True,
        min_delta=4,
        cooldown_up=10.0,
        cooldown_down=30.0,
    )
    assert changed
    assert cap == 92  # 8% step down

    cap2, last_adjust2, changed2 = openai_utils._smooth_wait_based_cap(
        current_cap=cap,
        candidate_cap=40,
        now=now + 5.0,
        last_adjust=last_adjust,
        limiter_pressure=True,
        min_delta=4,
        cooldown_up=10.0,
        cooldown_down=30.0,
    )
    assert not changed2
    assert cap2 == cap
    assert last_adjust2 == last_adjust


def test_wait_based_cap_allows_gentle_growth():
    now = time.time()
    cap, last_adjust, changed = openai_utils._smooth_wait_based_cap(
        current_cap=10,
        candidate_cap=30,
        now=now,
        last_adjust=0.0,
        limiter_pressure=True,
        min_delta=2,
        cooldown_up=0.0,
        cooldown_down=30.0,
    )
    assert changed
    assert cap == 12  # 18% step up from 10 (ceil)

    cap2, _, changed2 = openai_utils._smooth_wait_based_cap(
        current_cap=cap,
        candidate_cap=30,
        now=now + 1.0,
        last_adjust=last_adjust,
        limiter_pressure=True,
        min_delta=2,
        cooldown_up=0.0,
        cooldown_down=30.0,
    )
    assert changed2
    assert cap2 > cap


def test_rate_limit_decrement_is_aggressive():
    assert openai_utils._rate_limit_decrement(50) == 20
    assert openai_utils._rate_limit_decrement(3) == 3


def test_ramp_up_increases_parallelism(tmp_path):
    active = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def responder(prompt: str, **_: object):
        async with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        await asyncio.sleep(0.05)
        async with lock:
            active["current"] -= 1
        return [f"ok-{prompt}"], 0.01, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(12)],
            identifiers=[f"p{i}" for i in range(12)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=8,
            ramp_up_seconds=0.1,
            ramp_up_start_fraction=0.25,
            status_report_interval=None,
            logging_level="error",
        )
    )

    assert active["peak"] > 2


def test_ramp_up_worker_spawner_does_not_plateau_at_half_queue(tmp_path):
    active = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def responder(prompt: str, **_: object):
        async with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        await asyncio.sleep(0.20)
        async with lock:
            active["current"] -= 1
        return [f"ok-{prompt}"], 0.20, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(10)],
            identifiers=[f"p{i}" for i in range(10)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=10,
            ramp_up_seconds=0.05,
            ramp_up_start_fraction=0.2,
            status_report_interval=None,
            logging_level="error",
        )
    )

    assert active["peak"] >= 8


def test_ramp_up_halts_on_rate_limit(tmp_path, capsys):
    active = {"current": 0, "peak": 0}
    lock = asyncio.Lock()
    first_error = {"raised": False}

    async def responder(prompt: str, **_: object):
        async with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        if not first_error["raised"]:
            first_error["raised"] = True
            async with lock:
                active["current"] -= 1
            raise openai_utils.RateLimitError("rate limit")
        await asyncio.sleep(0.2)
        async with lock:
            active["current"] -= 1
        return [f"ok-{prompt}"], 0.01, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(10)],
            identifiers=[f"p{i}" for i in range(10)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=8,
            ramp_up_seconds=0.1,
            ramp_up_start_fraction=0.25,
            status_report_interval=None,
            global_cooldown=0,
            logging_level="error",
        )
    )

    output = capsys.readouterr().out
    assert "Halting ramp-up" in output
    assert active["peak"] <= 3


def test_ramp_up_does_not_halt_after_window(tmp_path, capsys):
    first_error = {"raised": False}

    async def responder(prompt: str, **_: object):
        if not first_error["raised"]:
            first_error["raised"] = True
            await asyncio.sleep(0.06)
            raise openai_utils.RateLimitError("rate limit")
        await asyncio.sleep(0.01)
        return [f"ok-{prompt}"], 0.01, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(4)],
            identifiers=[f"p{i}" for i in range(4)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=4,
            ramp_up_seconds=0.05,
            ramp_up_start_fraction=0.25,
            status_report_interval=None,
            global_cooldown=0,
            logging_level="error",
        )
    )

    output = capsys.readouterr().out
    assert "Halting ramp-up" not in output


def test_dynamic_timeout_uses_bounded_recent_window(tmp_path, monkeypatch):
    captured_lengths = []
    original = openai_utils._compute_dynamic_timeout_from_samples

    def wrapped(durations, *args, **kwargs):
        captured_lengths.append(len(list(durations)))
        return original(durations, *args, **kwargs)

    monkeypatch.setattr(openai_utils, "_compute_dynamic_timeout_from_samples", wrapped)

    async def responder(prompt: str, **_: object):
        await asyncio.sleep(0)
        return [f"ok-{prompt}"], 0.05, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(80)],
            identifiers=[f"p{i}" for i in range(80)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=True,
            max_timeout=1.0,
            timeout_factor=2.0,
            max_retries=1,
            n_parallels=4,
            ramp_up_seconds=0,
            status_report_interval=None,
            logging_level="error",
        )
    )

    assert captured_lengths
    assert max(captured_lengths) == openai_utils._dynamic_timeout_success_window(3)


def test_periodic_status_update_reports_p90_and_tps_in_requested_order(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_utils, "_ensure_runtime_dependencies", lambda **_: None)
    monkeypatch.setattr(
        openai_utils,
        "_get_rate_limit_headers",
        lambda *args, **kwargs: {
            "limit_requests": "30000",
            "remaining_requests": "30000",
            "limit_tokens": "180000000",
            "remaining_tokens": "180000000",
        },
    )

    async def fake_get_response(prompt: str, **_: object):
        await asyncio.sleep(0.15)
        raw = [
            {
                "id": f"resp-{prompt}",
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 30,
                    "output_tokens_details": {"reasoning_tokens": 20},
                },
                "output": [],
            }
        ]
        return [f"ok-{prompt}"], 0.20, raw

    monkeypatch.setattr(openai_utils, "get_response", fake_get_response)

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(18)],
            identifiers=[f"p{i}" for i in range(18)],
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=True,
            max_timeout=5.0,
            timeout_factor=2.0,
            max_retries=1,
            n_parallels=3,
            ramp_up_seconds=0,
            status_report_interval=0.2,
            print_example_prompt=False,
            logging_level="error",
        )
    )

    output = capsys.readouterr().out
    periodic_lines = [
        line for line in output.splitlines() if "Periodic status update:" in line
    ]
    assert periodic_lines
    line = periodic_lines[-1]
    pattern = re.compile(
        r"Periodic status update: "
        r"cost_so_far=.*?, "
        r"p90=\d+\.\d{2} sec, "
        r"timeouts=\d+/\d+(?: \(\+\d+ since last\))?, "
        r"rate_limit_errors=\d+/\d+(?: \(\+\d+ since last\))?, "
        r"connection_errors=\d+/\d+(?: \(\+\d+ since last\))?, "
        r"json_parse_errors=\d+/\d+(?: \(\+\d+ since last\))?, "
        r"tps=\d+\.\d{2}, "
        r"throughput<=\d+ prompts/min, "
        r"cap=\d+, active=\d+, inflight=\d+, awaiting_response=\d+, queue=\d+, processed=\d+/\d+"
    )
    assert pattern.search(line), line
