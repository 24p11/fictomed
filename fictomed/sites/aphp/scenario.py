"""Scenario builder — port of ``generate_scenario_from_profile`` to polars.

Given a *profile* row sampled from PMSI, produces a fully populated clinical
scenario dict ready to be turned into an LLM prompt by :mod:`prompt`. Cancer-
specific enrichment (TNM, biomarkers, treatment regimen) and secondary-
diagnosis sampling (chronic, metastases, complications, procedure) are all
done here.

Lookups (ICD/CCAM descriptions and synonyms) are pre-computed at context
build time so the per-profile loop stays cheap.

The original ``define_text_managment_type`` is **not** called from here —
it lives in :mod:`fictomed.sites.aphp.managment` and the pipeline orchestrator
chains the two so neither module depends on the other.
"""

from __future__ import annotations

import datetime as dt
import random as _random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import constants as C
from . import sampler

import math

# ---------------------------------------------------------------------------
# Context — collected DataFrames + lookup caches
# ---------------------------------------------------------------------------


@dataclass
class ScenarioContext:
    """Resolved tables and lookup caches needed to build scenarios.

    Built once per pipeline run via :func:`build_context`. Holds collected
    polars DataFrames (for sampling) and pre-computed Python dicts (for
    O(1) ICD/CCAM description lookups).
    """

    # Tables (collected, used for sampling)
    profiles: pl.DataFrame
    secondary_icd: pl.DataFrame  # already typed (Cancer/Metastasis/Chronic/Acute)
    procedures: pl.DataFrame
    cancer_treatment: pl.DataFrame
    names: pl.DataFrame
    hospitals: pl.DataFrame

    # Lookup caches
    icd_description: dict[str, str]
    procedure_description: dict[str, str]
    icd_synonyms_by_code: dict[str, list[str]]

    # Code sets used in conditional logic
    cancer_codes: frozenset[str]
    chronic_codes: frozenset[str]

    # Years pool for random date generation (defaults to last 3 calendar years)
    simulation_years: list[int] = field(
        default_factory=lambda: [
            dt.date.today().year - 2,
            dt.date.today().year - 1,
            dt.date.today().year,
        ]
    )


def _build_cancer_codes(cancer_codes_lf: pl.LazyFrame) -> frozenset[str]:
    """Reproduce the cancer-codes munging from utils_v2.py:218–224.

    Take the ``CIM10`` column, derive 3-character parent codes, union both,
    drop anything starting with ``Z``.
    """
    codes = cancer_codes_lf.select("CIM10").collect()["CIM10"].drop_nulls().to_list()
    parents = {c[:3] for c in codes}
    full = set(codes) | parents
    return frozenset(c for c in full if not c.startswith("Z"))


def _build_chronic_codes(
    chronic_lf: pl.LazyFrame, cancer_codes: frozenset[str]
) -> frozenset[str]:
    """Codes flagged chronic in [1,2,3], with cancer codes promoted to 3.

    Mirrors utils_v2.py:231–233.
    """
    df = chronic_lf.collect()
    promoted = df.with_columns(
        pl.when(pl.col("code").is_in(list(cancer_codes)))
        .then(pl.lit(3))
        .otherwise(pl.col("chronic"))
        .alias("chronic")
    )
    return frozenset(
        promoted.filter(pl.col("chronic").is_in([1, 2, 3]))["code"].to_list()
    )


def _build_icd_description(official_lf: pl.LazyFrame) -> dict[str, str]:
    """Build ``{icd_code: description}`` from the official ATIH dictionary."""
    df = official_lf.select(["icd_code", "icd_code_description"]).collect()
    return dict(zip(df["icd_code"].to_list(), df["icd_code_description"].to_list()))


def _build_procedure_description(proc_lf: pl.LazyFrame) -> dict[str, str]:
    """Build ``{procedure_code: description}`` from the CCAM dictionary."""
    df = proc_lf.select(["procedure", "procedure_description"]).collect()
    return dict(zip(df["procedure"].to_list(), df["procedure_description"].to_list()))


def _build_synonym_index(synonyms_lf: pl.LazyFrame) -> dict[str, list[str]]:
    """Group synonyms by ICD code: ``{code: [phrasing1, phrasing2, ...]}``."""
    df = synonyms_lf.select(["icd_code", "icd_code_description"]).collect()
    out: dict[str, list[str]] = {}
    for code, desc in zip(
        df["icd_code"].to_list(), df["icd_code_description"].to_list()
    ):
        out.setdefault(code, []).append(desc)
    return out


def _type_secondary_icd(
    secondary_lf: pl.LazyFrame,
    cancer_codes: frozenset[str],
    chronic_codes: frozenset[str],
) -> pl.DataFrame:
    """Tag each secondary-diagnosis row as Metastasis / Cancer / Chronic / Acute.

    Mirrors ``load_secondary_icd`` (utils_v2.py:438–442).
    """
    cancer = list(cancer_codes)
    chronic = list(chronic_codes)
    return secondary_lf.with_columns(
        pl.when(pl.col("icd_secondary_code").is_in(C.ICD_CANCER_META))
        .then(pl.lit("Metastasis"))
        .when(pl.col("icd_secondary_code").is_in(C.ICD_CANCER_META_LN))
        .then(pl.lit("Metastasis LN"))
        .when(pl.col("icd_secondary_code").is_in(cancer))
        .then(pl.lit("Cancer"))
        .when(pl.col("icd_secondary_code").is_in(chronic))
        .then(pl.lit("Chronic"))
        .otherwise(pl.lit("Acute"))
        .alias("type")
    ).collect()


def build_context(data: dict[str, pl.LazyFrame]) -> ScenarioContext:
    """Materialise everything needed to build scenarios from a *data* dict.

    *data* is the dictionary returned by :func:`fictomed.sites.aphp.loader.load_data`.
    """
    cancer_codes = _build_cancer_codes(data["cancer_codes"])
    chronic_codes = _build_chronic_codes(data["chronic"], cancer_codes)
    secondary_typed = _type_secondary_icd(
        data["secondary_icd"], cancer_codes, chronic_codes
    )

    return ScenarioContext(
        profiles=data["profiles"].collect(),
        secondary_icd=secondary_typed,
        procedures=data["procedures"].collect(),
        cancer_treatment=data["cancer_treatment"].collect(),
        names=data["names"].collect(),
        hospitals=data["hospitals"].collect(),
        icd_description=_build_icd_description(data["icd_official"]),
        procedure_description=_build_procedure_description(data["procedure_official"]),
        icd_synonyms_by_code=_build_synonym_index(data["icd_synonyms"]),
        cancer_codes=cancer_codes,
        chronic_codes=chronic_codes,
    )


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def lookup_icd_description(ctx: ScenarioContext, code: str) -> str:
    """Return the official ATIH ICD-10 description, or ``""`` if unknown."""
    return ctx.icd_description.get(code, "")


def lookup_procedure_description(ctx: ScenarioContext, code: str) -> str:
    """Return the official CCAM procedure description, or ``""`` if unknown."""
    return ctx.procedure_description.get(code, "")


def lookup_icd_synonym(
    ctx: ScenarioContext,
    code: str,
    rng: _random.Random | None = None,
) -> str:
    """Return one random alternative phrasing for *code*, falling back to official.

    For metastasis codes (``ICD_CANCER_META``), restrict to phrasings that
    contain ``"metastase"`` — mirrors ``get_icd_alternative_descriptions``
    (utils_v2.py:604).
    """
    rng = rng or _random
    pool = ctx.icd_synonyms_by_code.get(code, [])
    if code in C.ICD_CANCER_META:
        pool = [p for p in pool if "metastase" in p.lower()]
    if pool:
        return rng.choice(pool)
    return lookup_icd_description(ctx, code)


# ---------------------------------------------------------------------------
# Clinical scenario construction
# ---------------------------------------------------------------------------

_SCENARIO_KEYS: tuple[str, ...] = (
    "age",
    "sexe",
    "date_entry",
    "date_discharge",
    "date_of_birth",
    "first_name",
    "last_name",
    "icd_primary_code",
    "case_management_type",
    "icd_secondary_code",
    "text_secondary_icd_official",
    "procedure",
    "icd_primary_description",
    "admission_mode",
    "discharge_disposition",
    "cancer_stage",
    "score_TNM",
    "histological_type",
    "treatment_recommandation",
    "chemotherapy_regimen",
    "biomarkers",
    "department",
    "hospital",
    "first_name_med",
    "last_name_med",
    "template_name",
)

_GROUPING_SECONDARY_DEFAULT = [
    "icd_primary_code",
    "icd_secondary_code",
    "cage2",
    "sexe",
    "nb",
]
_GROUPING_SECONDARY_BY_DRG = [
    "drg_parent_code",
    "icd_secondary_code",
    "cage2",
    "sexe",
    "nb",
]
_GROUPING_PROCEDURE = [
    "procedure",
    "drg_parent_code",
    "icd_primary_code",
    "cage2",
    "sexe",
]


def _empty_scenario() -> dict[str, Any]:
    return {k: None for k in _SCENARIO_KEYS} | {
        "icd_secondary_code": [],
        "text_secondary_icd_official": "",
    }


def _format_secondary_line(code: str, description: str) -> str:
    return f"- {description} ({code})\n"


def _append_sampled_secondaries(
    scenario: dict[str, Any],
    sampled: pl.DataFrame,
    ctx: ScenarioContext,
) -> None:
    """Mutate *scenario* with the codes + formatted text from a sample batch."""
    if sampled.is_empty():
        return
    for row in sampled.iter_rows(named=True):
        code = row["icd_secondary_code"]
        desc = lookup_icd_description(ctx, code)
        scenario["text_secondary_icd_official"] += _format_secondary_line(code, desc)
    scenario["icd_secondary_code"].extend(sampled["icd_secondary_code"].to_list())

def _parse_secondary_codes(value: Any) -> list[str]:
    if value in (None, "", []):
        return []

    if isinstance(value, list):
        return [
            str(code).strip()
            for code in value
            if str(code).strip()
            and str(code).strip().upper() != "NA"
            and str(code).strip().lower() != "nan"
        ]

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.upper() == "NA" or value.lower() == "nan":
            return []

        return [
            code.strip()
            for code in value.replace(";", " ").split()
            if code.strip()
            and code.strip().upper() != "NA"
            and code.strip().lower() != "nan"
        ]

    value = str(value).strip()
    if value == "" or value.upper() == "NA" or value.lower() == "nan":
        return []

    return [value]


def build_scenario(
    ctx: ScenarioContext,
    profile: dict[str, Any],
    *,
    add_secondary: bool = False,
    rng: _random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Build one clinical-scenario dict from a single PMSI *profile* row.

    Mirrors ``generate_scenario_from_profile`` (utils_v2.py:942) step by
    step. Returns the scenario dict *without* the management-type / coding
    rule / template name fields — those are added by :mod:`managment` in
    a separate pass.
    """
    rng = rng or _random
    np_rng = np_rng or np.random.default_rng()

    profile = dict(profile)  # defensive copy

    if profile.get("sexe") is not None:
        profile["sexe"] = int(profile["sexe"])

    profile["icd_parent_code"] = (profile.get("icd_primary_code") or "")[:3]

    scenario = _empty_scenario()
    scenario.update(profile)

    age2 = profile.get("age2")

    if age2 is not None:
        scenario["age"] = int(age2)
    elif profile.get("cage") is not None:
        scenario["age"] = sampler.random_age(profile["cage"], rng=rng)
    else:
        raise ValueError(
            "Impossible de déterminer l'âge du patient : "
            "age2 est manquant et aucune colonne cage n'est disponible."
        )
    
    los = profile.get("los")
    los_mean = profile.get("los_mean")
    los_sd = profile.get("los_sd")
    year = rng.choice(ctx.simulation_years)

    if isinstance(los, float) and math.isnan(los):
        los = None

    admission_type = str(profile.get("admission_type") or "").strip().upper()
    admission_type = admission_type.replace("-", "_").replace(" ", "_")

    is_outpatient = admission_type in {
        "HP",
        "OUTPATIENT",
        "OUT_PATIENT",
        "AMBULATOIRE",
        "HOSPITALISATION_PARTIELLE",
        "HOSPITALISATION_DE_JOUR",
        "HDJ",
    }

    if is_outpatient:
        los = 0

    scenario["date_entry"], scenario["date_discharge"] = sampler.get_dates_of_stay(
        admission_type=profile.get("admission_type"),
        admission_mode=profile.get("admission_mode"),
        los_mean=los_mean,
        los_sd=los_sd,
        los=los,
        year=year,
        rng=rng,
        np_rng=np_rng,
    )
    if is_outpatient:
        scenario["date_discharge"] = scenario["date_entry"]
    age = scenario["age"] or 0
    scenario["date_of_birth"] = sampler.random_date_between(
        scenario["date_entry"] - dt.timedelta(days=365 * (age + 1)),
        scenario["date_entry"] - dt.timedelta(days=365 * age),
        rng=rng,
    )
    scenario["first_name"], scenario["last_name"] = sampler.pick_name(
        ctx.names, profile["sexe"], rng=rng
    )
    scenario["first_name_med"], scenario["last_name_med"] = sampler.pick_name(
        ctx.names, rng.randint(1, 2), rng=rng
    )
    scenario["department"] = profile.get("specialty")
    scenario["hospital"] = sampler.pick_hospital(ctx.hospitals, rng=rng)

    # --- Cancer-specific enrichment
    is_cancer = profile.get("icd_primary_code") in ctx.cancer_codes
    if profile.get("case_management_type") in {"DP", "Z511"}:
        treatment = sampler.sample_conditional(
            ctx.cancer_treatment, profile, nb=1, rng=rng, np_rng=np_rng
        )
        if not treatment.is_empty():
            row = treatment.row(0, named=True)
            scenario["histological_type"] = row.get("histological_type")
            tnm = row.get("TNM")
            if tnm not in (None, "Variable", "Non pertinent"):
                scenario["score_TNM"] = tnm
            stage = row.get("stage")
            if stage not in (None, "Variable", "Non pertinent"):
                scenario["cancer_stage"] = stage
            scenario["treatment_recommandation"] = row.get("treatment_recommandation")
            scenario["chemotherapy_regimen"] = row.get("chemotherapy_regimen")
            scenario["biomarkers"] = row.get("biomarkers")

    # --- Primary diagnosis descriptions
    scenario["icd_primary_description"] = lookup_icd_description(
        ctx, profile.get("icd_primary_code", "")
    )
    scenario["case_management_type_description"] = lookup_icd_description(
        ctx, profile.get("case_management_type", "")
    )

    # --- Secondary diagnoses already in profile
    secondary_codes = _parse_secondary_codes(profile.get("icd_secondary_code"))

    if secondary_codes:
        scenario["icd_secondary_code"] = secondary_codes
        for code in secondary_codes:
            scenario["text_secondary_icd_official"] += _format_secondary_line(
                code, lookup_icd_description(ctx, code)
            )

    if add_secondary:
        profile_for_secondary_sampling = {
            k: v for k, v in profile.items() if k != "icd_secondary_code"
        }
        _add_secondary_diagnoses(
            scenario, ctx, profile_for_secondary_sampling, is_cancer, rng=rng, np_rng=np_rng
        )

    # --- Procedure
    proc_pool = ctx.procedures.select(
        [c for c in _GROUPING_PROCEDURE if c in ctx.procedures.columns]
    )
    procedures = sampler.sample_conditional(
        proc_pool, profile, nb=1, rng=rng, np_rng=np_rng
    )
    if not procedures.is_empty():
        proc_code = procedures["procedure"][0]
        scenario["procedure"] = proc_code
        scenario["text_procedure"] = lookup_procedure_description(ctx, proc_code)
    else:
        scenario["procedure"] = ""
        scenario["text_procedure"] = ""

    return scenario


def _add_secondary_diagnoses(
    scenario: dict[str, Any],
    ctx: ScenarioContext,
    profile: dict[str, Any],
    is_cancer: bool,
    *,
    rng: _random.Random,
    np_rng: np.random.Generator,
) -> None:
    """Sample chronic / metastasis / complication diagnoses, in this order.

    Mirrors utils_v2.py:1037–1131. Mutates *scenario* in place.
    """
    # Reset accumulator (matches utils_v2 line 1037)
    scenario["text_secondary_icd_official"] = ""

    grouping = [
        c for c in _GROUPING_SECONDARY_DEFAULT if c in ctx.secondary_icd.columns
    ]

    # --- Chronic diseases (and cancer when not cancer-primary)
    if is_cancer:
        chronic_pool = ctx.secondary_icd.filter(pl.col("type") == "Chronic").select(
            grouping
        )
    else:
        chronic_pool = ctx.secondary_icd.filter(
            pl.col("type").is_in(["Chronic", "Cancer"])
        ).select(grouping)
    chronic = sampler.sample_conditional(
        chronic_pool, profile, distinct_chapter=True, rng=rng, np_rng=np_rng
    )
    _append_sampled_secondaries(scenario, chronic, ctx)

    # If we drew cancer codes during the chronic step, treat as cancer afterwards
    if scenario["icd_secondary_code"] and (
        set(scenario["icd_secondary_code"]) & ctx.cancer_codes
    ):
        is_cancer = True

    # --- Lymph-node and distant metastases (cancer scenarios only)
    if is_cancer:
        tnm = scenario.get("score_TNM")
        if tnm and not isinstance(tnm, float):
            if re.search(r"N[123x+]", tnm):
                pool = ctx.secondary_icd.filter(
                    pl.col("type") == "Metastasis LN"
                ).select(grouping)
                _append_sampled_secondaries(
                    scenario,
                    sampler.sample_conditional(
                        pool, profile, nb=1, rng=rng, np_rng=np_rng
                    ),
                    ctx,
                )
            if re.search(r"M[123x+]", tnm):
                pool = ctx.secondary_icd.filter(pl.col("type") == "Metastasis").select(
                    grouping
                )
                _append_sampled_secondaries(
                    scenario,
                    sampler.sample_conditional(pool, profile, rng=rng, np_rng=np_rng),
                    ctx,
                )
        else:
            pool = ctx.secondary_icd.filter(
                pl.col("type").is_in(["Metastasis", "Metastasis LN"])
            ).select(grouping)
            _append_sampled_secondaries(
                scenario,
                sampler.sample_conditional(pool, profile, rng=rng, np_rng=np_rng),
                ctx,
            )

    # --- Acute complications (grouped by drg_parent_code, not icd_primary_code)
    grouping_acute = [
        c for c in _GROUPING_SECONDARY_BY_DRG if c in ctx.secondary_icd.columns
    ]
    acute_pool = ctx.secondary_icd.filter(pl.col("type") == "Acute").select(
        grouping_acute
    )
    _append_sampled_secondaries(
        scenario,
        sampler.sample_conditional(acute_pool, profile, rng=rng, np_rng=np_rng),
        ctx,
    )


def format_aphp_scenario(
    df: pl.DataFrame,
    cancer_codes: frozenset[str],
    atih_rules: dict[str, dict],
    prompt_workflow: str = "one_stage",
) -> pl.DataFrame:
    """AP-HP-specific scenario formatting.

    Adds:
    - scenario
    - user_prompt
    - system_prompt
    - prefix
    - prefix_len
    """
    from fictomed.sites.aphp import prompt

    user_prompts: list[str] = []
    if prompt_workflow not in {"one_stage", "two_stage"}:
        raise ValueError(
            "prompt_workflow doit valoir 'one_stage' ou 'two_stage'"
        )

    system_prompts: list[str] = []
    prefixes: list[str] = []
    summary_system_prompts: list[str] = []
    final_system_prompts: list[str] = []

    for row in df.iter_rows(named=True):
        user_prompt = prompt.make_user_prompt(row, cancer_codes, atih_rules)
        prefix = prompt.make_prefix(row, cancer_codes)

        user_prompts.append(user_prompt)
        prefixes.append(prefix)
        if prompt_workflow == "one_stage":
            system_prompts.append(prompt.load_system_prompt(row["template_name"]))
        else:
            summary_system_prompts.append(
                prompt.load_system_prompt(row["template_name"], stage="summary")
            )
            final_system_prompts.append(
                prompt.load_system_prompt(row["template_name"], stage="final")
            )

    common_columns = [
        pl.Series("scenario", user_prompts, dtype=pl.Utf8),
        pl.Series("user_prompt", user_prompts, dtype=pl.Utf8),
    ]
    if prompt_workflow == "one_stage":
        return df.with_columns(
            *common_columns,
            pl.Series("system_prompt", system_prompts, dtype=pl.Utf8),
            pl.Series("prefix", prefixes, dtype=pl.Utf8),
            pl.Series("prefix_len", [len(p) for p in prefixes], dtype=pl.Int64),
        )

    return df.with_columns(
        *common_columns,
        pl.Series("summary_system_prompt", summary_system_prompts, dtype=pl.Utf8),
        pl.Series("summary_prefix", [""] * df.height, dtype=pl.Utf8),
        pl.Series("final_system_prompt", final_system_prompts, dtype=pl.Utf8),
        pl.Series("final_prefix", prefixes, dtype=pl.Utf8),
    )
