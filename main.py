import argparse

from fictomed import PIPELINES, generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génération de séjours fictifs et scénarios médicaux."
    )
    parser.add_argument(
        "pipeline",
        choices=list(PIPELINES.keys()),
        help="Nom du pipeline à utiliser (ex: brest, rennes...)",
    )
    parser.add_argument("--n-sejours", type=int, default=1000)
    parser.add_argument("--n-ccam", type=int, default=1)
    parser.add_argument("--n-das", type=int, default=5)
    parser.add_argument("--ghm5-pattern", type=str, default=None)
    parser.add_argument(
        "--prompt-workflow",
        choices=["one_stage", "two_stage"],
        default="one_stage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate(
        pipeline_name=args.pipeline,
        n_sejours=args.n_sejours,
        n_ccam=args.n_ccam,
        n_das=args.n_das,
        ghm5_pattern=args.ghm5_pattern,
        prompt_workflow=args.prompt_workflow,
    )
    print(f"✓ {len(df)} séjours générés avec le pipeline '{args.pipeline}'")


if __name__ == "__main__":
    main()
