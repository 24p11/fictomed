from fictomed.config import load_config
from fictomed.registry import CONFIG_DIR, PIPELINES
from pathlib import Path
from datetime import datetime


def generate(
    pipeline_name: str,
    n_sejours: int = 1000,
    n_ccam: int = 1,
    n_das: int = 5,
    ghm5_pattern: str | None = None,
    config_file: str | None = None,
    prompt_workflow: str = "one_stage",
) -> None:
    """Orchestrate an end-to-end synthetic medical-report generation run."""
    if pipeline_name not in PIPELINES:
        raise ValueError(
            f"Pipeline inconnu : '{pipeline_name}'. "
            f"Valeurs acceptées : {list(PIPELINES.keys())}"
        )
    if config_file:
        config = load_config(config_file)
    else:
        config = load_config(CONFIG_DIR / "servers.yaml")

    pipeline = PIPELINES[pipeline_name](
        config=config["pipelines"][pipeline_name],
        prompt_workflow=prompt_workflow,
    )

    pipeline.check_data()
    data = pipeline.load_data()

    df = pipeline.get_fictive(
        data,
        n_sejours=n_sejours,
        n_ccam=n_ccam,
        n_das=n_das,
        ghm5_pattern=ghm5_pattern,
    )
    df = pipeline.get_scenario(df)

    output_dir = pipeline.config.get("data", {}).get("output")
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / (
            f"{pipeline_name}_scenarios_{df.height}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        )

        df.write_parquet(output_path)
        print(f"Scénarios sauvegardés dans {output_path}")

    return df
