"""User-prompt formatter and system-prompt loader for the AP-HP pipeline.

Ports ``make_prompts_marks_from_scenario`` (utils_v2.py:1162) and the
template-loading part of ``create_system_prompt`` (utils_v2.py:1263).

The orchestrator calls:
1. :func:`make_user_prompt` after building the scenario + management decision
   to produce the text block that replaces ``[SCENARIO here]`` in the template.
2. :func:`load_system_prompt` to read the chosen ``.txt`` template file.

The scenario dict must already contain the management fields added by
:mod:`fictomed.sites.aphp.managment`:
- ``case_management_type_text``          — French description of the hospitalisation context
- ``coding_rule``    — ATIH rule id (e.g. ``"T1"``, ``"D3-2"``, ``"other"``)
- ``template_name``  — filename of the system-prompt template
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from .loader import TEMPLATES_DIR

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_templates_dir() -> Path:
    """Temporary eval helper.

    Allows switching AP-HP templates with:
    FICTOMED_APHP_TEMPLATE_SET=template_v1
    FICTOMED_APHP_TEMPLATE_SET=template_v2
    """
    base_dir = Path(__file__).resolve().parent / "templates"

    template_set = os.getenv("FICTOMED_APHP_TEMPLATE_SET")
    if template_set:
        return base_dir / template_set

    return base_dir

def _interpret_sexe(sexe: int | str | None) -> str:
    """Return ``"Masculin"`` / ``"Féminin"`` from the PMSI gender code."""
    if sexe is None:
        return ""
    sexe = int(sexe)
    return "Masculin" if sexe == 1 else "Féminin"


def _fmt_date(d: dt.date | None) -> str | None:
    return d.strftime("%d/%m/%Y") if d is not None else None

def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        # gère NaN éventuel
        if value != value:
            return False
    except Exception:
        pass
    return str(value).strip() != ""


def _format_groupage_ghm(scenario: dict[str, Any]) -> str:
    """Formatte le groupage PMSI/GHM pour le scénario donné au LLM.

    Attention : drg_parent_code correspond ici à la racine de GHM.
    """
    drg_parent_code = scenario.get("drg_parent_code")
    drg_parent_description = scenario.get("drg_parent_description")

    if not _has_value(drg_parent_code) and not _has_value(drg_parent_description):
        return ""

    text = "- Groupage PMSI / GHM :\n"

    if _has_value(drg_parent_code):
        text += f"   * Racine de GHM : {drg_parent_code}\n"

    if _has_value(drg_parent_description):
        text += f"   * Libellé de la racine de GHM : {drg_parent_description}\n"


    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_user_prompt(
    scenario: dict[str, Any],
    cancer_codes: frozenset[str],
    atih_rules: dict[str, dict],
) -> str:
    """Build the ``**SCÉNARIO DE DÉPART :**`` text block.

    Mirrors ``make_prompts_marks_from_scenario`` (utils_v2.py:1162) step by
    step. The original iterates over ``scenario.items()`` and fires multiple
    ``if`` blocks per key; this port reproduces the same ordering and
    conditions explicitly so clinical content is identical.

    Parameters
    ----------
    scenario:
        Scenario dict enriched with the management fields ``case_management_type_text``,
        ``coding_rule``, and ``template_name`` (added by the orchestrator
        after calling :func:`fictomed.sites.aphp.managment.define_managment_type`).
    cancer_codes:
        Frozenset from :class:`fictomed.sites.aphp.scenario.ScenarioContext`.
    atih_rules:
        Dict from :func:`fictomed.sites.aphp.loader.load_atih_rules`.
    """
    is_cancer = scenario.get("icd_primary_code") in cancer_codes
    coding_rule = scenario.get("coding_rule", "")

    # Look up the ATIH rule description (shown in the hospitalisation context line)
    case_management_description = (
        atih_rules[coding_rule]["texte"] if coding_rule in atih_rules else ""
    )

    SCENARIO = "**SCÉNARIO DE DÉPART :**\n"

    # Age
    age = scenario.get("age")
    if age is not None:
        SCENARIO += "- Âge du patient : " + str(age) + " ans\n"

    # Sexe
    sexe = scenario.get("sexe")
    if sexe is not None:
        SCENARIO += "- Sexe du patient : " + _interpret_sexe(sexe) + "\n"

    # Dates of stay and birth
    date_entry = scenario.get("date_entry")
    if date_entry is not None:
        SCENARIO += "- Date d'entrée : " + _fmt_date(date_entry) + "\n"

    date_discharge = scenario.get("date_discharge")
    if date_discharge is not None:
        SCENARIO += "- Date de sortie : " + _fmt_date(date_discharge) + "\n"

    date_of_birth = scenario.get("date_of_birth")
    if date_of_birth is not None:
        SCENARIO += "- Date de naissance : " + _fmt_date(date_of_birth) + "\n"

    # Patient identity
    last_name = scenario.get("last_name")
    if last_name is not None:
        SCENARIO += "- Nom du patient : " + last_name + "\n"

    first_name = scenario.get("first_name")
    if first_name is not None:
        SCENARIO += "- Prénom du patient : " + first_name + "\n"

    # Cancer-specific fields (mirroring the ``icd_codes_cancer`` guard at 1189)
    if is_cancer:
        icd_primary_description = scenario.get("icd_primary_description")
        icd_primary_code = scenario.get("icd_primary_code", "")
        if icd_primary_description is not None:
            SCENARIO += (
                "- Localisation anatomique de la tumeur primaire : "
                + icd_primary_description
                + " ("
                + icd_primary_code
                + ")\n"
            )

        histological_type = scenario.get("histological_type")
        if histological_type is not None and not isinstance(histological_type, float):
            SCENARIO += (
                "- Type anatomopathologique de la tumeur primaire : "
                + histological_type
                + "\n"
            )
        else:
            SCENARIO += (
                "- Type anatomopathologique de la tumeur primaire : Vous choisirez "
                "vous même un type histologique cohérent avec la localisation anatomique\n"
            )

        score_TNM = scenario.get("score_TNM")
        if score_TNM is not None and not isinstance(score_TNM, float):
            SCENARIO += "- Score TNM : " + score_TNM + "\n"
        else:
            SCENARIO += (
                "- Score TNM : Si la notion de score de TNM est pertinente avec le "
                "type histologique et la localisation anatomique, vous choisirez un score TNM\n"
            )

        cancer_stage = scenario.get("cancer_stage")
        if cancer_stage is not None and not isinstance(cancer_stage, float):
            SCENARIO += "- Stade tumoral : " + cancer_stage + "\n"

        biomarkers = scenario.get("biomarkers")
        if biomarkers is not None and not isinstance(biomarkers, float):
            SCENARIO += "- Biomarqueurs tumoraux : " + biomarkers + "\n"
        else:
            SCENARIO += (
                "- Biomarqueurs tumoraux : Vous choisirez des biomarqueurs tumoraux "
                "cohérents avec la localisation anatomique et l'histologie de la tumeur\n"
            )

    # Admission / discharge modes
    admission_mode = scenario.get("admission_mode")
    if admission_mode is not None:
        SCENARIO += "- Mode d'entrée' : " + str(admission_mode) + "\n"

    discharge_disposition = scenario.get("discharge_disposition")
    if discharge_disposition is not None:
        SCENARIO += "- Mode de sortie' : " + str(discharge_disposition) + "\n"

    # Hospitalisation context + ICD coding block
    case_management_type = scenario.get("case_management_type")
    if case_management_type is not None:
        #SCENARIO += _format_groupage_ghm(scenario)

        case_management_type_text = scenario.get("case_management_type_text", "")
        situa = scenario.get("situa", "")
        
        if situa :
            SCENARIO += (
                "- Contexte de l'hospitalisation : "
                + situa
                + ".\n"
            )
        else :
            SCENARIO += (
                "- Contexte de l'hospitalisation : "
                + "non disponible"
                + ".\n"
            )
        SCENARIO += "- Codage CIM10 :\n"
        SCENARIO += (
            "   * Diagnostic principal : "
            + (scenario.get("icd_primary_description") or "")
            + " ("
            + (scenario.get("icd_primary_code") or "")
            + ")\n"
        )
        text_secondary_icd = (scenario.get("text_secondary_icd_official") or "").strip()

        SCENARIO += "   * Diagnostics associés :\n"

        if text_secondary_icd:
            for line in text_secondary_icd.splitlines():
                line = line.strip()
                if line:
                    SCENARIO += f"      - {line.lstrip('- ').strip()}\n"
        else:
            SCENARIO += "      - aucun\n"

    # Surgical / interventional procedure (only for C or K type DRGs)
    drg_parent_code = scenario.get("drg_parent_code") or ""
    procedure = scenario.get("procedure")
    if procedure is not None and drg_parent_code[2:3] in ("C", "K"):
        SCENARIO += (
            "- Acte CCAM : "
            + (scenario.get("text_procedure") or "").lower() +" (" + (scenario.get("procedure") or "") + ")\n"
        )

    # Physician identity
    first_name_med = scenario.get("first_name_med")
    if first_name_med is not None:
        SCENARIO += (
            "- Nom du médecin / signataire : "
            + first_name_med
            + " "
            + (scenario.get("last_name_med") or "")
            + "\n"
        )

    # Department / specialty — stored as ``department`` in the Stream scenario dict
    # (the original used a ``specialty`` key, utils_v2.py:1235)
    department = scenario.get("department") or scenario.get("specialty")
    if department is not None and not isinstance(department, float):
        SCENARIO += "- Service : " + str(department) + "\n"

    hospital = scenario.get("hospital")
    if hospital is not None and not isinstance(hospital, float):
        SCENARIO += "- Hôpital : " + str(hospital) + "\n"

    # Cancer instructions block (appended after the main fields)
    if is_cancer:
        SCENARIO += "Ce cas clinique concerne un patient présentant un cancer\n"
        histological_type = scenario.get("histological_type")
        if histological_type is not None:
            SCENARIO += (
                "Vous choisirez un épisode de traitement sachant que les recommandations "
                "pour ce stade du cancer sont les suivantes :\n"
            )
            SCENARIO += (
                "   - Schéma thérapeutique : "
                + (scenario.get("treatment_recommandation") or "")
                + "\n"
            )
            chemo = scenario.get("chemotherapy_regimen")
            if chemo is not None and not isinstance(chemo, float):
                SCENARIO += "   - Protocole de chimiothérapie : " + chemo + "\n"

        SCENARIO += (
            "Veillez à bien préciser le type histologique et la valeur des "
            "biomarqueurs si recherchés\n"
        )

    return SCENARIO

def make_prefix(
    scenario: dict[str, Any],
    cancer_codes: frozenset[str],
) -> str:
    """Build the assistant prefix used by the historical AP-HP Mistral workflow."""
    if scenario.get("icd_primary_code") in cancer_codes:
        return """Le compte rendu suivant respecte les élements suivants :
        - les diagnostics ont une formulation moins formelle que la définition du code
        - le type histologique et la valeur des biomarqueurs si recherchés
        - le plan du CRH est conforme aux recommandations.
        """

    return """Le compte rendu suivant respecte les élements suivants :
        - les diagnostics ont une formulation moins formelle que la définition du code
        - le plan du CRH est conforme aux recommandations.
        """


def make_final_user_prompt(base_user_prompt: str, summary: str) -> str:
    """Append the intermediate clinical summary for the final generation."""
    return (
        base_user_prompt.rstrip()
        + "\n\n**RÉSUMÉ CLINIQUE CONTRÔLÉ :**\n"
        + summary.strip()
        + "\n"
    )


def load_system_prompt(template_name: str, stage: str | None = None) -> str:
    """Read and return the content of a system-prompt template file.

    Mirrors the file-read in ``create_system_prompt`` (utils_v2.py:1284).
    Templates live in ``fictomed/sites/aphp/templates/`` and, for staged
    workflows, in its ``summary`` and ``final`` subdirectories.

    Parameters
    ----------
    template_name:
        Filename only (e.g. ``"medical_inpatient.txt"``), as set by
        :func:`fictomed.sites.aphp.managment.define_managment_type`.
    """
    if stage not in {None, "summary", "final"}:
        raise ValueError("stage doit valoir None, 'summary' ou 'final'")

    if stage is None:
        templates_dir = _get_templates_dir()
    else:
        templates_dir = Path(__file__).resolve().parent / "templates" / stage

    filename = str(template_name)
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    path = templates_dir / filename

    with path.open("r", encoding="utf-8") as f:
        return f.read()
