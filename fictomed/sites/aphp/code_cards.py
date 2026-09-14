from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_MISSING_VALUES = {"", "NA", "NAN", "NONE", "NULL"}
_CODE_RE = re.compile(r"\b[A-Z][0-9]{2}\.?[0-9A-Z]*\b")


def normalize_icd_code(value: Any) -> str:
    """Return a canonical ICD-10 code.

    Examples
    --------
    A014 -> A01.4
    A01.4 -> A01.4
    C50. -> C50
    C50 -> C50
    D839 -> D83.9
    """
    if value is None:
        return ""

    code = str(value).strip().upper()
    code = code.strip("()[]{};:,")
    code = code.replace(" ", "")
    code = code.rstrip(".")
    code = code.rstrip("+*")

    if code in _MISSING_VALUES:
        return ""

    if not re.fullmatch(r"[A-Z][0-9]{2}\.?[0-9A-Z]*", code):
        return ""

    if "." in code:
        return code

    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"

    return code


def category_code(value: Any) -> str:
    code = normalize_icd_code(value)
    if len(code) < 3:
        return ""
    return code[:3]


def _read_index(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []

    text = index_path.read_text(encoding="utf-8-sig")
    sample = text[:4096]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        dialect = csv.excel_tab

    rows: list[dict[str, str]] = []
    for row in csv.DictReader(text.splitlines(), dialect=dialect):
        rows.append({str(k): str(v) for k, v in row.items() if k is not None})
    return rows


def _extract_codes_from_value(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        codes: list[str] = []
        for item in value:
            codes.extend(_extract_codes_from_value(item))
        return codes

    text = str(value).strip()
    if text.upper() in _MISSING_VALUES:
        return []

    codes = []
    for match in _CODE_RE.findall(text.upper()):
        code = normalize_icd_code(match)
        if code:
            codes.append(code)
    return codes


def extract_icd_codes_from_scenario(row: Mapping[str, Any]) -> list[str]:
    """Extract ICD-10 codes from a generated AP-HP scenario row.

    Prefer structured fields. Fallback to text_secondary_icd_official only for DAS.
    """
    ordered_keys = [
        # DP
        "icd_primary_code",
        "diagnostic_principal_code",
        "primary_icd_code",
        "dp",
        # DR / MDP if present
        "icd_related_code",
        "related_icd_code",
        "diagnostic_relie_code",
        "dr",
        "mdp",
        # DAS
        "icd_secondary_code",
        "secondary_icd_codes",
        "diagnostic_associes",
        "diagnostics_associes",
        "das",
        "secondary_diagnoses",
    ]

    codes: list[str] = []

    for key in ordered_keys:
        if key in row:
            codes.extend(_extract_codes_from_value(row.get(key)))

    if not codes and "text_secondary_icd_official" in row:
        codes.extend(_extract_codes_from_value(row.get("text_secondary_icd_official")))

    deduped: list[str] = []
    seen: set[str] = set()

    for code in codes:
        if code and code not in seen:
            deduped.append(code)
            seen.add(code)

    return deduped


@dataclass(frozen=True)
class CodeCard:
    input_code: str
    card_code: str
    card_type: str
    path: Path
    text: str


@dataclass(frozen=True)
class CodeCardsResult:
    codes: list[str]
    found_codes: list[str]
    missing_codes: list[str]
    text: str


@dataclass
class CodeCardsRegistry:
    exact_dir: Path
    category_dir: Path
    exact_index: dict[str, Path]
    category_index: dict[str, Path]

    @classmethod
    def from_dirs(
        cls,
        exact_dir: Path,
        category_dir: Path,
    ) -> CodeCardsRegistry:
        exact_index: dict[str, Path] = {}
        category_index: dict[str, Path] = {}

        # Contrat recode-icd (docs/livraison/CONTRAT.md, format_version 1) :
        # la colonne canonique du chemin est `fichier` ; `filepath` est le
        # nom historique, accepté en repli pour les bibliothèques
        # antérieures au contrat.
        for row in _read_index(exact_dir / "index.csv"):
            code = normalize_icd_code(row.get("code"))
            filepath = row.get("fichier") or row.get("filepath")
            if code and filepath:
                exact_index[code] = exact_dir / filepath

        for row in _read_index(category_dir / "index.csv"):
            code = category_code(row.get("code"))
            filepath = row.get("fichier") or row.get("filepath")
            if code and filepath:
                category_index[code] = category_dir / filepath

        return cls(
            exact_dir=exact_dir,
            category_dir=category_dir,
            exact_index=exact_index,
            category_index=category_index,
        )

    def find(self, code: str) -> CodeCard | None:
        normalized = normalize_icd_code(code)
        if not normalized:
            return None

        exact_path = self.exact_index.get(normalized)
        if exact_path and exact_path.exists():
            return CodeCard(
                input_code=code,
                card_code=normalized,
                card_type="code",
                path=exact_path,
                text=exact_path.read_text(encoding="utf-8"),
            )

        cat = category_code(normalized)
        category_path = self.category_index.get(cat)
        if cat and category_path and category_path.exists():
            return CodeCard(
                input_code=code,
                card_code=cat,
                card_type="category",
                path=category_path,
                text=category_path.read_text(encoding="utf-8"),
            )

        return None

    def build_prompt_section(self, codes: Iterable[str]) -> CodeCardsResult:
        found: list[CodeCard] = []
        missing: list[str] = []

        seen_cards: set[tuple[str, str]] = set()
        normalized_codes: list[str] = []

        for raw_code in codes:
            code = normalize_icd_code(raw_code)
            if not code:
                continue

            normalized_codes.append(code)
            card = self.find(code)

            if card is None:
                missing.append(code)
                continue

            key = (card.card_type, card.card_code)
            if key not in seen_cards:
                found.append(card)
                seen_cards.add(key)

        if not found:
            return CodeCardsResult(
                codes=normalized_codes,
                found_codes=[],
                missing_codes=missing,
                text="",
            )

        parts = [
            "**FICHES DESCRIPTIVES DES CODES CIM-10 DU SCÉNARIO :**",
            "",
            "Les fiches ci-dessous précisent le périmètre des codes CIM-10 présents dans le scénario.",
            "Elles doivent être utilisées pour choisir des formulations compatibles avec les codes, sans ajouter de diagnostic non codé.",
            "",
        ]

        for card in found:
            parts.append(card.text.strip())
            parts.append("")

        return CodeCardsResult(
            codes=normalized_codes,
            found_codes=[card.card_code for card in found],
            missing_codes=missing,
            text="\n".join(parts).strip(),
        )


def append_code_cards_to_user_prompt(
    user_prompt: str,
    cards_text: str,
) -> str:
    if not cards_text:
        return user_prompt
    return user_prompt.rstrip() + "\n\n" + cards_text.strip() + "\n"


def add_code_cards_to_row(
    row: Mapping[str, Any],
    registry: CodeCardsRegistry,
) -> dict[str, Any]:
    out = dict(row)

    codes = extract_icd_codes_from_scenario(out)
    result = registry.build_prompt_section(codes)

    out["code_cards_codes"] = json.dumps(result.codes, ensure_ascii=False)
    out["code_cards_found_codes"] = json.dumps(result.found_codes, ensure_ascii=False)
    out["code_cards_missing_codes"] = json.dumps(
        result.missing_codes,
        ensure_ascii=False,
    )
    out["code_cards_text"] = result.text

    if result.text and out.get("user_prompt"):
        out["user_prompt"] = append_code_cards_to_user_prompt(
            str(out["user_prompt"]),
            result.text,
        )

    return out