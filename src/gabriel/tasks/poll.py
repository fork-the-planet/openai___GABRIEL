from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from gabriel.core.prompt_template import PromptTemplate, resolve_template
from gabriel.tasks.seed import Seed, SeedConfig
from gabriel.utils.file_utils import save_dataframe_with_fallback
from gabriel.utils.logging import announce_prompt_rendering
from gabriel.utils.openai_utils import get_all_responses, response_to_text
from gabriel.utils.parsing import safe_json
from ._run_utils import (
    hash_identifier,
    load_run_metadata,
    resolve_identifier_hash_bits,
    write_task_run_metadata,
)


@dataclass
class PollConfig:
    population_description: Optional[str] = None
    questions: Optional[Sequence[str]] = None
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "poll_results.csv"
    seed_file_name: str = "poll_seeds.csv"
    persona_file_name: str = "poll_personas.csv"
    seed_model: str = "gpt-5.5"
    persona_model: str = "gpt-5.5"
    poll_model: str = "gpt-5.5"
    n_parallels: int = 650
    num_personas: int = 1000
    entities_per_generation: int = 50
    entity_batch_frac: float = 0.25
    existing_entities_cap: int = 100
    deduplicate: bool = False
    deduplicate_sample_seed: int = 42
    n_questions_per_run: int = 8
    seed_additional_instructions: Optional[str] = None
    additional_instructions: Optional[str] = None
    web_search: bool = False
    reasoning_effort: Optional[str] = None
    use_dummy: bool = False
    seed_column_name: str = "seed"
    persona_column_name: str = "persona"

    def __post_init__(self) -> None:
        for field_name in (
            "population_description",
            "seed_additional_instructions",
            "additional_instructions",
            "reasoning_effort",
            "seed_column_name",
            "persona_column_name",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            cleaned = str(value).strip()
            setattr(self, field_name, cleaned or None)

        if self.population_description == "":
            self.population_description = None
        if not isinstance(self.num_personas, int) or self.num_personas <= 0:
            raise ValueError("num_personas must be a positive integer")
        if (
            not isinstance(self.entities_per_generation, int)
            or self.entities_per_generation <= 0
        ):
            raise ValueError("entities_per_generation must be a positive integer")
        if not 0 < float(self.entity_batch_frac) <= 1:
            raise ValueError("entity_batch_frac must be between 0 and 1")
        if (
            not isinstance(self.existing_entities_cap, int)
            or self.existing_entities_cap < 0
        ):
            raise ValueError("existing_entities_cap must be a non-negative integer")
        if (
            not isinstance(self.n_questions_per_run, int)
            or self.n_questions_per_run <= 0
        ):
            raise ValueError("n_questions_per_run must be a positive integer")

        normalized_questions: List[str] = []
        if self.questions is None:
            self.questions = []
        elif isinstance(self.questions, str):
            self.questions = [self.questions]
        else:
            self.questions = list(self.questions)

        for raw_question in self.questions:
            question = str(raw_question).strip()
            if not question:
                continue
            normalized_questions.append(question)

        if len(set(normalized_questions)) != len(normalized_questions):
            raise ValueError("questions must be unique after trimming whitespace")

        self.questions = normalized_questions


class Poll:
    """Generate representative synthetic respondents and survey them."""

    def __init__(
        self,
        cfg: PollConfig,
        *,
        persona_template: Optional[PromptTemplate] = None,
        persona_template_path: Optional[str] = None,
        answer_template: Optional[PromptTemplate] = None,
        answer_template_path: Optional[str] = None,
        seed_template_path: Optional[str] = None,
    ) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.seed_template_path = seed_template_path
        self.persona_template = resolve_template(
            template=persona_template,
            template_path=persona_template_path,
            reference_filename="persona_prompt.jinja2",
        )
        self.answer_template = resolve_template(
            template=answer_template,
            template_path=answer_template_path,
            reference_filename="poll_prompt.jinja2",
        )

    async def run(
        self,
        df: Optional[pd.DataFrame] = None,
        column_name: Optional[str] = None,
        *,
        reset_files: bool = False,
        response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
        embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
        get_all_embeddings_fn: Optional[
            Callable[..., Awaitable[Dict[str, List[float]]]]
        ] = None,
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        questions = list(self.cfg.questions or [])
        if self._has_existing_personas(df):
            persona_df = self._prepare_existing_persona_df(
                df=df,
                column_name=column_name,
            )
        else:
            base_df = await self._prepare_population(
                df=df,
                column_name=column_name,
                reset_files=reset_files,
                response_fn=response_fn,
                get_all_responses_fn=get_all_responses_fn,
                embedding_fn=embedding_fn,
                get_all_embeddings_fn=get_all_embeddings_fn,
                **response_kwargs,
            )

            persona_df = await self._generate_personas(
                base_df,
                reset_files=reset_files,
                response_fn=response_fn,
                get_all_responses_fn=get_all_responses_fn,
                **response_kwargs,
            )

        questions_to_answer = self._filter_existing_question_columns(
            persona_df,
            questions,
            reset_files=reset_files,
        )

        if not questions_to_answer:
            final_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
            save_dataframe_with_fallback(
                persona_df,
                final_path,
                index=False,
                label="Poll",
            )
            if questions:
                print(
                    "[Poll] All requested question columns already exist and "
                    "reset_files=False; returning the provided data unchanged."
                )
            else:
                print("[Poll] No questions provided; returning seeds and personas only.")
            return persona_df

        return await self._answer_questions(
            persona_df,
            questions_to_answer,
            reset_files=reset_files,
            response_fn=response_fn,
            get_all_responses_fn=get_all_responses_fn,
            **response_kwargs,
        )

    def _prepare_existing_persona_df(
        self,
        *,
        df: Optional[pd.DataFrame],
        column_name: Optional[str],
    ) -> pd.DataFrame:
        if df is None:
            raise ValueError("df must be provided when reusing existing personas")

        print(
            "[Poll] Found existing 'persona' column in provided DataFrame; "
            "skipping seeding and persona generation."
        )
        out_df = df.reset_index(drop=True).copy()
        out_df[self.cfg.persona_column_name] = self._coerce_nonempty_texts(
            out_df[self.cfg.persona_column_name].tolist(),
            self.cfg.persona_column_name,
        )

        if (
            self.cfg.seed_column_name not in out_df.columns
            and column_name
            and column_name in out_df.columns
            and column_name != self.cfg.persona_column_name
        ):
            out_df[self.cfg.seed_column_name] = self._coerce_nonempty_texts(
                out_df[column_name].tolist(),
                column_name,
            )

        persona_path = os.path.join(self.cfg.save_dir, self.cfg.persona_file_name)
        save_dataframe_with_fallback(
            out_df,
            persona_path,
            index=False,
            label="Poll",
        )
        return out_df

    async def _prepare_population(
        self,
        *,
        df: Optional[pd.DataFrame],
        column_name: Optional[str],
        reset_files: bool,
        response_fn: Optional[Callable[..., Awaitable[Any]]],
        get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]],
        embedding_fn: Optional[Callable[..., Awaitable[Any]]],
        get_all_embeddings_fn: Optional[
            Callable[..., Awaitable[Dict[str, List[float]]]]
        ],
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        if df is not None:
            source_column = self._resolve_seed_column(df, column_name)
            print(
                f"[Poll] Using provided DataFrame column '{source_column}' as respondent seeds."
            )
            base_df = df.reset_index(drop=True).copy()
            base_df[self.cfg.seed_column_name] = self._coerce_nonempty_texts(
                base_df[source_column].tolist(),
                source_column,
            )
            if "seed_id" not in base_df.columns:
                if "entity_id" in base_df.columns:
                    base_df["seed_id"] = base_df["entity_id"].astype(str)
                else:
                    base_df["seed_id"] = [
                        f"seed-{idx:05d}" for idx in range(len(base_df))
                    ]
            final_seed_path = os.path.join(self.cfg.save_dir, self.cfg.seed_file_name)
            save_dataframe_with_fallback(
                base_df,
                final_seed_path,
                index=False,
                label="Poll",
            )
            return base_df

        if not self.cfg.population_description:
            raise ValueError(
                "Provide either population_description or df with a seed column."
            )

        print(
            f"[Poll] Seeding {self.cfg.num_personas} respondents from population description."
        )
        seed_cfg = SeedConfig(
            instructions=self._build_seed_instructions(self.cfg.population_description),
            save_dir=self.cfg.save_dir,
            file_name=self.cfg.seed_file_name,
            model=self.cfg.seed_model,
            n_parallels=self.cfg.n_parallels,
            num_entities=self.cfg.num_personas,
            entities_per_generation=self.cfg.entities_per_generation,
            entity_batch_frac=self.cfg.entity_batch_frac,
            existing_entities_cap=self.cfg.existing_entities_cap,
            deduplicate=self.cfg.deduplicate,
            deduplicate_sample_seed=self.cfg.deduplicate_sample_seed,
            reasoning_effort=self.cfg.reasoning_effort,
            use_dummy=self.cfg.use_dummy,
        )
        seed_task = Seed(seed_cfg, template_path=self.seed_template_path)
        seed_df = await seed_task.run(
            existing_entities=None,
            reset_files=reset_files,
            response_fn=response_fn,
            get_all_responses_fn=get_all_responses_fn,
            embedding_fn=embedding_fn,
            get_all_embeddings_fn=get_all_embeddings_fn,
            **response_kwargs,
        )
        base_df = seed_df.rename(
            columns={
                "entity": self.cfg.seed_column_name,
                "entity_id": "seed_id",
            }
        ).copy()
        base_df[self.cfg.seed_column_name] = self._coerce_nonempty_texts(
            base_df[self.cfg.seed_column_name].tolist(),
            self.cfg.seed_column_name,
        )
        final_seed_path = os.path.join(self.cfg.save_dir, self.cfg.seed_file_name)
        save_dataframe_with_fallback(
            base_df,
            final_seed_path,
            index=False,
            label="Poll",
        )
        return base_df

    async def _generate_personas(
        self,
        df: pd.DataFrame,
        *,
        reset_files: bool,
        response_fn: Optional[Callable[..., Awaitable[Any]]],
        get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]],
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        out_df = df.reset_index(drop=True).copy()
        if out_df.empty:
            out_df[self.cfg.persona_column_name] = pd.Series(dtype="object")
            persona_path = os.path.join(self.cfg.save_dir, self.cfg.persona_file_name)
            save_dataframe_with_fallback(
                out_df,
                persona_path,
                index=False,
                label="Poll",
            )
            return out_df

        seed_texts = self._coerce_nonempty_texts(
            out_df[self.cfg.seed_column_name].tolist(),
            self.cfg.seed_column_name,
        )
        base_ids, row_ids, id_to_text = self._build_unique_text_index(seed_texts)
        prompts: List[str] = []
        identifiers: List[str] = []
        announce_prompt_rendering("Poll", len(base_ids))
        print(f"[Poll] Generating personas for {len(base_ids)} unique seeds.")
        for ident in base_ids:
            identifiers.append(ident)
            prompts.append(
                self.persona_template.render(
                    seed=id_to_text[ident],
                    population_description=self.cfg.population_description,
                )
            )

        raw_path = os.path.join(self.cfg.save_dir, "poll_personas_raw_responses.csv")
        df_resp = await self._dispatch_responses(
            prompts=prompts,
            identifiers=identifiers,
            model=self.cfg.persona_model,
            json_mode=False,
            save_path=raw_path,
            reset_files=reset_files,
            response_fn=response_fn,
            get_all_responses_fn=get_all_responses_fn,
            **response_kwargs,
        )
        persona_lookup = {
            str(identifier): response_to_text(raw).strip()
            for identifier, raw in zip(df_resp["Identifier"], df_resp["Response"])
        }
        out_df[self.cfg.persona_column_name] = [
            persona_lookup.get(ident, "") for ident in row_ids
        ]

        persona_path = os.path.join(self.cfg.save_dir, self.cfg.persona_file_name)
        save_dataframe_with_fallback(
            out_df,
            persona_path,
            index=False,
            label="Poll",
        )
        return out_df

    async def _answer_questions(
        self,
        df: pd.DataFrame,
        questions: Sequence[str],
        *,
        reset_files: bool,
        response_fn: Optional[Callable[..., Awaitable[Any]]],
        get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]],
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        out_df = df.reset_index(drop=True).copy()
        if out_df.empty:
            for question in questions:
                out_df[question] = pd.Series(dtype="object")
            final_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
            save_dataframe_with_fallback(
                out_df,
                final_path,
                index=False,
                label="Poll",
            )
            return out_df

        question_batches = [
            list(questions[i : i + self.cfg.n_questions_per_run])
            for i in range(0, len(questions), self.cfg.n_questions_per_run)
        ]
        if len(question_batches) > 1:
            print(
                f"[Poll] {len(questions)} questions provided. "
                f"n_questions_per_run={self.cfg.n_questions_per_run}. "
                f"Splitting into {len(question_batches)} prompt batches."
            )

        if self.cfg.seed_column_name in out_df.columns:
            seed_texts = self._coerce_nonempty_texts(
                out_df[self.cfg.seed_column_name].tolist(),
                self.cfg.seed_column_name,
            )
        else:
            seed_texts = [""] * len(out_df)
        persona_texts = self._coerce_nonempty_texts(
            out_df[self.cfg.persona_column_name].tolist(),
            self.cfg.persona_column_name,
        )
        combined = [
            f"{seed}\n\n<persona>\n{persona}" if seed else f"<persona>\n{persona}"
            for seed, persona in zip(seed_texts, persona_texts)
        ]
        raw_path = os.path.join(self.cfg.save_dir, "poll_answers_raw_responses.csv")
        run_metadata = load_run_metadata(
            self.cfg.save_dir, "poll_answers", reset_files=reset_files
        )
        identifier_hash_bits = resolve_identifier_hash_bits(
            task_name="Poll",
            metadata=run_metadata,
            reset_files=reset_files,
            checkpoint_paths=[raw_path],
        )
        write_task_run_metadata(
            save_dir=self.cfg.save_dir,
            base_name="poll_answers",
            task_name="Poll",
            model=self.cfg.poll_model,
            identifier_hash_bits=identifier_hash_bits,
            n_attributes_per_run=None,
            attribute_batches=[],
        )
        base_ids, row_ids, _ = self._build_unique_text_index(
            combined,
            identifier_hash_bits=identifier_hash_bits,
        )
        first_rows: Dict[str, int] = {}
        for row_idx, ident in enumerate(row_ids):
            first_rows.setdefault(ident, row_idx)

        prompts: List[str] = []
        identifiers: List[str] = []
        announce_prompt_rendering("Poll", len(base_ids) * len(question_batches))
        print(
            f"[Poll] Answering {len(questions)} question(s) for {len(base_ids)} unique personas."
        )
        for batch_idx, batch in enumerate(question_batches):
            for ident in base_ids:
                row_idx = first_rows[ident]
                identifiers.append(f"{ident}_batch{batch_idx}")
                prompts.append(
                    self.answer_template.render(
                        persona=persona_texts[row_idx],
                        questions=self._format_questions(batch),
                        additional_instructions=self.cfg.additional_instructions,
                    )
                )

        df_resp = await self._dispatch_responses(
            prompts=prompts,
            identifiers=identifiers,
            model=self.cfg.poll_model,
            json_mode=True,
            save_path=raw_path,
            reset_files=reset_files,
            response_fn=response_fn,
            get_all_responses_fn=get_all_responses_fn,
            **response_kwargs,
        )
        response_lookup = dict(zip(df_resp["Identifier"], df_resp["Response"]))

        answers_by_id: Dict[str, Dict[str, Any]] = {ident: {} for ident in base_ids}
        for batch_idx, batch in enumerate(question_batches):
            for ident in base_ids:
                parsed = self._parse_answers(
                    response_lookup.get(f"{ident}_batch{batch_idx}"),
                    batch,
                    sample_key=ident,
                )
                answers_by_id[ident].update(parsed)

        for question in questions:
            out_df[question] = [answers_by_id.get(ident, {}).get(question) for ident in row_ids]

        final_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        save_dataframe_with_fallback(
            out_df,
            final_path,
            index=False,
            label="Poll",
        )
        return out_df

    async def _dispatch_responses(
        self,
        *,
        prompts: List[str],
        identifiers: List[str],
        model: str,
        json_mode: bool,
        save_path: str,
        reset_files: bool,
        response_fn: Optional[Callable[..., Awaitable[Any]]],
        get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]],
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        driver = get_all_responses_fn or get_all_responses
        kwargs = self._filter_response_kwargs(response_kwargs)
        if response_fn is not None:
            kwargs.setdefault("response_fn", response_fn)
        if json_mode and "web_search" not in kwargs:
            kwargs["web_search"] = self.cfg.web_search
        df_resp = await driver(
            prompts=prompts,
            identifiers=identifiers,
            save_path=save_path,
            model=model,
            json_mode=json_mode,
            n_parallels=self.cfg.n_parallels,
            reset_files=reset_files,
            reasoning_effort=self.cfg.reasoning_effort,
            use_dummy=self.cfg.use_dummy,
            **kwargs,
        )
        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")
        return df_resp

    def _build_seed_instructions(self, population_description: str) -> str:
        instructions = f"""
Each entity is a compact but richly informative demographic/personality seed for one plausible real person drawn from the following population:
{population_description}

Each seed should be 2-4 sentences and should include concrete details that would help a later persona generator branch in meaningfully different directions: age or life stage, location and setting, education/work/family situation, class position, religion or lack thereof when relevant, language/community context, and a few realistic behavioral, preference, upbringing, personality, or circumstantial details.

The overall set must be representative of the described population rather than idealized or performatively diverse. Statistical realism matters, but so does within-group variation. Include enough specific texture that later personas do not collapse together. Remember outlier people exist occasionally, and everyone is a complex individual with at least a few unique (but realistic) traits.
Enforce realism and representativeness across all persona dimensions, and allow for the occasional odd duck.
"""
        if self.cfg.seed_additional_instructions:
            instructions = (
                instructions.strip()
                + "\n\nAdditional researcher instructions:\n"
                + self.cfg.seed_additional_instructions
            )
        return instructions.strip()

    @staticmethod
    def _filter_response_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        filtered = dict(kwargs)
        filtered.pop("embedding_fn", None)
        filtered.pop("get_all_embeddings_fn", None)
        return filtered

    @staticmethod
    def _format_questions(questions: Sequence[str]) -> str:
        return "\n".join(f"- {question}" for question in questions)

    @staticmethod
    def _normalize_key(text: Any) -> str:
        return " ".join(str(text).strip().split())

    def _parse_answers(
        self,
        raw: Any,
        questions: Sequence[str],
        *,
        sample_key: str,
    ) -> Dict[str, Any]:
        parsed_obj = safe_json(response_to_text(raw))
        result: Dict[str, Any] = {question: None for question in questions}
        if isinstance(parsed_obj, dict):
            used_keys: set[str] = set()
            normalized_obj = {
                self._normalize_key(key): key for key in parsed_obj.keys()
            }
            for question in questions:
                exact_key = normalized_obj.get(self._normalize_key(question))
                if exact_key is None:
                    continue
                result[question] = self._select_answer_from_value(
                    parsed_obj.get(exact_key),
                    question=question,
                    sample_key=sample_key,
                )
                used_keys.add(exact_key)

            if len(questions) == 1 and result[questions[0]] is None:
                for fallback_key in ("output", "answer", "response"):
                    exact_key = normalized_obj.get(fallback_key)
                    if exact_key is None:
                        continue
                    result[questions[0]] = self._select_answer_from_value(
                        parsed_obj.get(exact_key),
                        question=questions[0],
                        sample_key=sample_key,
                    )
                    used_keys.add(exact_key)
                    break

            missing_questions = [q for q, value in result.items() if value is None]
            if missing_questions:
                unused_values = [
                    value
                    for key, value in parsed_obj.items()
                    if key not in used_keys
                ]
                if len(unused_values) == len(missing_questions):
                    for question, value in zip(missing_questions, unused_values):
                        result[question] = self._select_answer_from_value(
                            value,
                            question=question,
                            sample_key=sample_key,
                        )

            return result

        if len(questions) == 1:
            scalar = self._select_answer_from_value(
                parsed_obj if parsed_obj is not None else response_to_text(raw),
                question=questions[0],
                sample_key=sample_key,
            )
            result[questions[0]] = scalar
        return result

    def _select_answer_from_value(
        self,
        value: Any,
        *,
        question: str,
        sample_key: str,
    ) -> Any:
        distribution = self._coerce_distribution(value)
        if distribution:
            return self._sample_from_distribution(
                distribution,
                question=question,
                sample_key=sample_key,
            )
        if isinstance(value, str):
            return self._parse_scalar_value(value)
        return value

    def _sample_from_distribution(
        self,
        distribution: Sequence[Tuple[Any, float]],
        *,
        question: str,
        sample_key: str,
    ) -> Any:
        total = sum(prob for _, prob in distribution)
        if total <= 0:
            return distribution[0][0]

        draw = self._stable_uniform(sample_key=sample_key, question=question)
        cumulative = 0.0
        normalized = [(answer, prob / total) for answer, prob in distribution]
        for answer, prob in normalized:
            cumulative += prob
            if draw <= cumulative or math.isclose(cumulative, 1.0):
                return answer
        return normalized[-1][0]

    @classmethod
    def _coerce_distribution(cls, value: Any) -> List[Tuple[Any, float]]:
        if isinstance(value, dict):
            nested = value.get("options") or value.get("answers") or value.get("distribution")
            if isinstance(nested, list):
                return cls._coerce_distribution(nested)
            return []
        if not isinstance(value, list):
            return []

        distribution: List[Tuple[Any, float]] = []
        for item in value:
            answer: Any
            probability: Any
            if isinstance(item, dict):
                if "answer" not in item or "probability" not in item:
                    continue
                answer = item.get("answer")
                probability = item.get("probability")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                answer, probability = item[0], item[1]
            else:
                continue

            try:
                prob = float(probability)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(prob) or prob < 0:
                continue
            distribution.append((cls._coerce_answer_value(answer), prob))
        return distribution

    @classmethod
    def _coerce_answer_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._parse_scalar_value(value)
        return value

    @staticmethod
    def _stable_uniform(*, sample_key: str, question: str) -> float:
        digest = hashlib.sha1(f"{sample_key}|{question}".encode("utf-8")).digest()
        numerator = int.from_bytes(digest[:8], "big")
        return numerator / float(1 << 64)

    @staticmethod
    def _parse_scalar_value(text: str) -> Any:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except Exception:
            return cleaned

    @staticmethod
    def _build_unique_text_index(
        values: Sequence[str],
        *,
        identifier_hash_bits: int = 64,
    ) -> tuple[List[str], List[str], Dict[str, str]]:
        base_ids: List[str] = []
        row_ids: List[str] = []
        id_to_text: Dict[str, str] = {}
        seen: set[str] = set()
        for text in values:
            ident = hash_identifier(text, bits=identifier_hash_bits)
            row_ids.append(ident)
            if ident in seen:
                continue
            seen.add(ident)
            base_ids.append(ident)
            id_to_text[ident] = text
        return base_ids, row_ids, id_to_text

    @staticmethod
    def _coerce_nonempty_texts(values: Sequence[Any], column_name: str) -> List[str]:
        cleaned: List[str] = []
        missing_rows: List[int] = []
        for idx, value in enumerate(values):
            text = "" if value is None else str(value).strip()
            if not text:
                missing_rows.append(idx)
            cleaned.append(text)
        if missing_rows:
            preview = ", ".join(str(idx) for idx in missing_rows[:10])
            raise ValueError(
                f"Column '{column_name}' contains empty values at row(s): {preview}"
            )
        return cleaned

    def _resolve_seed_column(
        self,
        df: pd.DataFrame,
        column_name: Optional[str],
    ) -> str:
        if column_name is not None:
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame")
            return column_name
        for candidate in (self.cfg.seed_column_name, "entity"):
            if candidate in df.columns:
                return candidate
        if len(df.columns) == 1:
            return str(df.columns[0])
        raise ValueError(
            "column_name must be provided when df does not contain a clear seed column."
        )

    def _filter_existing_question_columns(
        self,
        df: pd.DataFrame,
        questions: Sequence[str],
        *,
        reset_files: bool,
    ) -> List[str]:
        if reset_files:
            return list(questions)

        existing = [question for question in questions if question in df.columns]
        if existing:
            print(
                "[Poll] Skipping "
                f"{len(existing)} question(s) already present in the DataFrame "
                "because reset_files=False."
            )
            for question in existing:
                print(f"[Poll] Reusing existing answers for: {question}")
        return [question for question in questions if question not in df.columns]

    @staticmethod
    def _has_existing_personas(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and "persona" in df.columns
