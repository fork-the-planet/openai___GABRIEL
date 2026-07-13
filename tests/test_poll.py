import asyncio
import json
from typing import Any, Dict, List

import pandas as pd
import pytest

import gabriel


def test_poll_with_population_description_batches_questions_and_returns_columns(
    monkeypatch,
    tmp_path,
):
    captured_seed_kwargs: Dict[str, Any] = {}
    stage_calls: List[Dict[str, Any]] = []

    async def fake_seed_run(self, **kwargs):
        captured_seed_kwargs.update(kwargs)
        return pd.DataFrame(
            {
                "entity": [
                    "Young Phoenix retail worker living with cousins and juggling community college.",
                    "Retired small-town widower in Alabama on a fixed income and active in church.",
                ],
                "entity_id": ["entity-00000", "entity-00001"],
                "source_batch": [0, 0],
                "source_identifier": ["seed", "seed"],
            }
        )

    async def fake_get_all_responses(
        *,
        prompts,
        identifiers,
        save_path,
        json_mode,
        **kwargs,
    ):
        stage_calls.append(
            {
                "save_path": save_path,
                "json_mode": json_mode,
                "max_output_tokens": kwargs.get("max_output_tokens"),
                "web_search": kwargs.get("web_search"),
                "identifiers": list(identifiers),
            }
        )
        responses: List[str] = []
        if save_path.endswith("poll_personas_raw_responses.csv"):
            for ident in identifiers:
                responses.append(
                    "\n".join(
                        [
                            f"Name: Persona {ident}",
                            "Age: 41",
                            "Location: Somewhere in the United States",
                            "Background: Grounded demographic background.",
                            "Daily Life: Busy and specific daily routine.",
                            "Life Story: A plausible personal history.",
                            "Worldview: Mixed, practical, and imperfect.",
                            "Personality: Strengths and flaws both present.",
                            "Goals and Fears: Wants stability and worries about decline.",
                            "Speech Style: Plainspoken and direct.",
                        ]
                    )
                )
        else:
            for prompt in prompts:
                payload: Dict[str, Any] = {}
                if "Rate support from 1 to 7 using an integer only." in prompt:
                    payload["Rate support from 1 to 7 using an integer only."] = [
                        [5, 1.0]
                    ]
                if "Return true or false only: would you vote for this policy?" in prompt:
                    payload[
                        "Return true or false only: would you vote for this policy?"
                    ] = [[True, 1.0], [False, 0.0]]
                if "In English only, briefly explain why." in prompt:
                    payload["In English only, briefly explain why."] = [
                        ["It seems fair.", 1.0]
                    ]
                responses.append(json.dumps(payload))
        return pd.DataFrame({"Identifier": identifiers, "Response": responses})

    monkeypatch.setattr("gabriel.tasks.poll.Seed.run", fake_seed_run)

    with pytest.warns(FutureWarning, match="deprecated and ignored") as caught:
        result = asyncio.run(
            gabriel.poll(
                population_description="a representative sample of the United States population",
                questions=[
                    "Rate support from 1 to 7 using an integer only.",
                    "Return true or false only: would you vote for this policy?",
                    "In English only, briefly explain why.",
                ],
                save_dir=str(tmp_path / "poll"),
                num_personas=2,
                n_questions_per_run=2,
                max_output_tokens=123,
                web_search=True,
                get_all_responses_fn=fake_get_all_responses,
            )
        )

    assert list(result["seed"]) == [
        "Young Phoenix retail worker living with cousins and juggling community college.",
        "Retired small-town widower in Alabama on a fixed income and active in church.",
    ]
    assert "persona" in result.columns
    assert result["Rate support from 1 to 7 using an integer only."].tolist() == [5, 5]
    assert result[
        "Return true or false only: would you vote for this policy?"
    ].tolist() == [True, True]
    assert result["In English only, briefly explain why."].tolist() == [
        "It seems fair.",
        "It seems fair.",
    ]
    assert captured_seed_kwargs["reset_files"] is False
    assert len(stage_calls) == 2
    assert len(caught) == 1
    assert stage_calls[0]["json_mode"] is False
    assert stage_calls[1]["json_mode"] is True
    assert stage_calls[0]["max_output_tokens"] is None
    assert stage_calls[1]["max_output_tokens"] is None
    assert stage_calls[0]["web_search"] is None
    assert stage_calls[1]["web_search"] is True
    assert len(stage_calls[1]["identifiers"]) == 4


def test_poll_uses_provided_dataframe_and_skips_seed(monkeypatch, tmp_path):
    async def unexpected_seed_run(self, **kwargs):
        raise AssertionError("Seed.run should not be called when df is provided")

    async def fake_get_all_responses(
        *,
        prompts,
        identifiers,
        save_path,
        json_mode,
        **kwargs,
    ):
        assert save_path.endswith("poll_personas_raw_responses.csv")
        responses = []
        for ident in identifiers:
            responses.append(
                "\n".join(
                    [
                        f"Name: Persona {ident}",
                        "Age: 29",
                        "Location: Test City",
                        "Background: Test background.",
                        "Daily Life: Test routine.",
                        "Life Story: Test story.",
                        "Worldview: Test worldview.",
                        "Personality: Test personality.",
                        "Goals and Fears: Test goals.",
                        "Speech Style: Test style.",
                    ]
                )
            )
        return pd.DataFrame({"Identifier": identifiers, "Response": responses})

    monkeypatch.setattr("gabriel.tasks.poll.Seed.run", unexpected_seed_run)

    df = pd.DataFrame(
        {
            "demographics": [
                "Urban bilingual service worker in Los Angeles.",
                "Suburban engineer raising two children in Ohio.",
            ]
        }
    )
    result = asyncio.run(
        gabriel.poll(
            df=df,
            column_name="demographics",
            save_dir=str(tmp_path / "poll_df"),
            get_all_responses_fn=fake_get_all_responses,
        )
    )

    assert list(result["seed"]) == list(df["demographics"])
    assert "persona" in result.columns
    assert "seed_id" in result.columns


def test_poll_api_passes_embedding_overrides_to_seed(monkeypatch, tmp_path):
    captured_seed_kwargs: Dict[str, Any] = {}

    async def fake_seed_run(self, **kwargs):
        captured_seed_kwargs.update(kwargs)
        return pd.DataFrame(
            {
                "entity": ["Test respondent seed."],
                "entity_id": ["entity-00000"],
                "source_batch": [0],
                "source_identifier": ["seed"],
            }
        )

    async def fake_get_all_responses(
        *,
        prompts,
        identifiers,
        save_path,
        json_mode,
        **kwargs,
    ):
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [
                    "\n".join(
                        [
                            "Name: Test Persona",
                            "Age: 33",
                            "Location: Testville",
                            "Background: Test background.",
                            "Daily Life: Test routine.",
                            "Life Story: Test story.",
                            "Worldview: Test worldview.",
                            "Personality: Test personality.",
                            "Goals and Fears: Test goals.",
                            "Speech Style: Test style.",
                        ]
                    )
                ],
            }
        )

    async def custom_embedding(text: str):
        return [float(len(text))]

    async def custom_embedding_driver(texts, identifiers):
        return {ident: [float(idx)] for idx, ident in enumerate(identifiers)}

    monkeypatch.setattr("gabriel.tasks.poll.Seed.run", fake_seed_run)

    result = asyncio.run(
        gabriel.poll(
            population_description="test population",
            save_dir=str(tmp_path / "poll_embeddings"),
            num_personas=1,
            embedding_fn=custom_embedding,
            get_all_embeddings_fn=custom_embedding_driver,
            get_all_responses_fn=fake_get_all_responses,
        )
    )

    assert len(result) == 1
    assert captured_seed_kwargs["embedding_fn"] is custom_embedding
    assert captured_seed_kwargs["get_all_embeddings_fn"] is custom_embedding_driver


def test_poll_reuses_existing_personas_and_skips_seed_and_persona_generation(
    monkeypatch,
    tmp_path,
):
    async def unexpected_seed_run(self, **kwargs):
        raise AssertionError("Seed.run should not be called when personas already exist")

    async def fake_get_all_responses(
        *,
        prompts,
        identifiers,
        save_path,
        json_mode,
        **kwargs,
    ):
        assert save_path.endswith("poll_answers_raw_responses.csv")
        assert json_mode is True
        assert len(prompts) == 2
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [
                    json.dumps(
                        {
                            "Return true or false only: Is this a reuse test?": [
                                [True, 1.0],
                                [False, 0.0],
                            ],
                        }
                    )
                    for _ in identifiers
                ],
            }
        )

    monkeypatch.setattr("gabriel.tasks.poll.Seed.run", unexpected_seed_run)

    df = pd.DataFrame(
        {
            "persona": [
                "Name: Persona A\nAge: 40\nBasic Political Views: Mixed\nLife Goals: Stability",
                "Name: Persona B\nAge: 28\nBasic Political Views: Mixed\nLife Goals: Growth",
            ]
        }
    )
    result = asyncio.run(
        gabriel.poll(
            df=df,
            save_dir=str(tmp_path / "poll_existing_personas"),
            questions=["Return true or false only: Is this a reuse test?"],
            get_all_responses_fn=fake_get_all_responses,
        )
    )

    assert result["persona"].tolist() == df["persona"].tolist()
    assert result["Return true or false only: Is this a reuse test?"].tolist() == [
        True,
        True,
    ]


def test_poll_skips_existing_question_columns_unless_reset_files(monkeypatch, tmp_path):
    async def unexpected_seed_run(self, **kwargs):
        raise AssertionError("Seed.run should not be called when personas already exist")

    captured_prompts: List[str] = []

    async def fake_get_all_responses(
        *,
        prompts,
        identifiers,
        save_path,
        json_mode,
        **kwargs,
    ):
        captured_prompts.extend(prompts)
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [
                    json.dumps({"Question B": [["new answer", 1.0]]})
                    for _ in identifiers
                ],
            }
        )

    monkeypatch.setattr("gabriel.tasks.poll.Seed.run", unexpected_seed_run)

    df = pd.DataFrame(
        {
            "persona": [
                "Name: Persona A\nAge: 40\nBasic Political Views: Mixed\nLife Goals: Stability",
            ],
            "Question A": ["existing answer"],
        }
    )
    result = asyncio.run(
        gabriel.poll(
            df=df,
            save_dir=str(tmp_path / "poll_existing_questions"),
            questions=["Question A", "Question B"],
            get_all_responses_fn=fake_get_all_responses,
        )
    )

    assert result["Question A"].tolist() == ["existing answer"]
    assert result["Question B"].tolist() == ["new answer"]
    assert len(captured_prompts) == 1
    assert "Question A" not in captured_prompts[0]
    assert "Question B" in captured_prompts[0]

    captured_prompts.clear()
    regenerated = asyncio.run(
        gabriel.poll(
            df=df,
            save_dir=str(tmp_path / "poll_existing_questions_reset"),
            questions=["Question A", "Question B"],
            reset_files=True,
            get_all_responses_fn=fake_get_all_responses,
        )
    )

    assert regenerated["Question A"].tolist() == [None]
    assert regenerated["Question B"].tolist() == ["new answer"]
    assert len(captured_prompts) == 1
    assert "Question A" in captured_prompts[0]
    assert "Question B" in captured_prompts[0]


def test_poll_samples_from_probability_distribution_deterministically():
    poll = gabriel.tasks.poll.Poll(gabriel.tasks.poll.PollConfig(save_dir="/tmp"))

    first = poll._select_answer_from_value(
        [["alpha", 0.2], ["beta", 0.8]],
        question="Question",
        sample_key="persona-1",
    )
    second = poll._select_answer_from_value(
        [["alpha", 0.2], ["beta", 0.8]],
        question="Question",
        sample_key="persona-1",
    )
    third = poll._select_answer_from_value(
        [["alpha", 1.0], ["beta", 0.0]],
        question="Question",
        sample_key="persona-2",
    )

    assert first == second
    assert third == "alpha"
