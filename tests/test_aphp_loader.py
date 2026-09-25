"""Tests du loader AP-HP (``fictomed.sites.aphp.loader``).

Le dépôt n'a pas de suite de tests configurée : ce fichier est exécutable tel
quel (``python tests/test_aphp_loader.py``) et reste compatible pytest
(``pytest tests/``) le jour où une suite est mise en place.

Il faut un interpréteur où ``fictomed`` et ``polars`` sont importables, par
exemple le venv d'un projet qui installe ce clone en éditable.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import polars as pl

from fictomed.sites.aphp import loader as L


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path, mtime: float, content: bytes = b"") -> Path:
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))
    return path


def _write_parquet(path: Path, df: pl.DataFrame, mtime: float) -> Path:
    df.write_parquet(path)
    os.utime(path, (mtime, mtime))
    return path


def _old_format_profile() -> pl.DataFrame:
    """Ancien format (``scenarios_bn_all_*.pq``) : ``age`` = classe d'âge,
    ``agean`` = âge exact, ``n`` = effectif."""
    return pl.DataFrame(
        {
            "racine": ["01C03"],
            "age": ["ge_18"],
            "agean": [42],
            "n": [7],
            "duree": [3],
            "mode_hospit": ["H"],
        }
    )


def _c1_format_profile() -> pl.DataFrame:
    """Format campagne C1+ : ``cage`` = classe d'âge déjà nommée, ``age`` porte
    autre chose (pivot / âge exact) et doit rester intact."""
    return pl.DataFrame(
        {
            "racine": ["01C03"],
            "age": ["42"],
            "cage": ["[40-50["],
            "duree": [3],
            "mode_hospit": ["H"],
        }
    )


# ---------------------------------------------------------------------------
# Sélection du fichier de profils
# ---------------------------------------------------------------------------


def test_profiles_selection_ignores_newer_txt_report() -> None:
    """Un rapport ``scenarios_*.txt`` plus récent que le ``.parquet`` ne doit
    jamais être choisi comme fichier de profils."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t0 = time.time() - 1000
        _write_parquet(d / "scenarios_C1.parquet", _c1_format_profile(), t0)
        _write_parquet(d / "scenarios_C1_dp.parquet", _c1_format_profile(), t0 + 10)
        # rapport texte déposé juste après le parquet (cas réel : +4 ms)
        _touch(d / "scenarios_C1_dp.rapport.txt", t0 + 10.004, b"rapport")
        chosen = L._resolve_pmsi_file(d, "profiles")
        assert chosen.name == "scenarios_C1_dp.parquet", chosen


def test_profiles_selection_keeps_legacy_pq_extension() -> None:
    """L'extension historique ``.pq`` reste reconnue, et le plus récent gagne
    quelle que soit l'extension."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t0 = time.time() - 1000
        _write_parquet(d / "scenarios_C1.parquet", _c1_format_profile(), t0)
        _write_parquet(d / "scenarios_bn_all_20260128.pq", _old_format_profile(), t0 + 10)
        _touch(d / "scenarios_C1.rapport.txt", t0 + 20, b"rapport")
        assert L._resolve_pmsi_file(d, "profiles").name == "scenarios_bn_all_20260128.pq"

        # seul un .pq : toujours trouvé
        (d / "scenarios_C1.parquet").unlink()
        assert L._resolve_pmsi_file(d, "profiles").name == "scenarios_bn_all_20260128.pq"


def test_profiles_selection_txt_only_raises() -> None:
    """Sans parquet/pq, un rapport texte seul ne doit pas être pris par défaut."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _touch(d / "scenarios_C1_dp.rapport.txt", time.time(), b"rapport")
        try:
            L._resolve_pmsi_file(d, "profiles")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("un rapport .txt seul a été accepté comme profil")


def test_all_pmsi_patterns_carry_an_extension() -> None:
    """Garde-fou : aucun motif nu (sans extension) dans ``_PMSI_PATTERNS``."""
    for key, patterns in L._PMSI_PATTERNS.items():
        for pat in patterns:
            suffix = Path(pat).suffix
            assert suffix and "*" not in suffix, f"motif sans extension pour {key!r}: {pat!r}"


# ---------------------------------------------------------------------------
# Invariant des mappings de renommage
# ---------------------------------------------------------------------------


def test_rename_mappings_have_no_source_as_target() -> None:
    """Le filtre « cible absente du schéma » de ``_safe_rename`` n'est correct
    que si aucune cible n'est aussi une source (sinon une colonne renommée
    pourrait bloquer un autre renommage). Ce test fige cette hypothèse."""
    for name in ("_PROFILE_RENAME", "_SECONDARY_RENAME"):
        mapping: dict[str, str] = getattr(L, name)
        sources, targets = set(mapping), set(mapping.values())
        assert not (sources & targets), f"{name}: cible(s) aussi source(s) : {sources & targets}"
        assert len(targets) == len(mapping), f"{name}: cibles dupliquées"


# ---------------------------------------------------------------------------
# Exécution directe (sans pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests ok")
    sys.exit(1 if failed else 0)
