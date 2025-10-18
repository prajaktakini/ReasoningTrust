# python
from omegaconf import DictConfig
import hydra
import logging

# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     # basic logging setup using config value
#     level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)
#     logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
#     logging.info("Starting reasoning_trust app")
#     logging.info("Config: %s", cfg.pretty())
#     # TODO: replace with real entrypoint logic
#     # For example: runner = Runner(cfg); runner.run()
#     print("Run completed")



# python
import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_toxicity(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    max_samples: int,
    file_name: str,
    template_id: int,
    model_name: str,
    enable_thinking: bool,
):
    """
    Import and invoke the toxicity pipeline main() function.
    Import is done inside the function to avoid startup cost when not used.
    """
    try:
        from reasoning_trust.perspectives.toxicity import process_data
    except Exception as e:
        logger.error("Failed to import toxicity pipeline: %s", e)
        raise

    logger.info("Starting toxicity pipeline: dataset=%s split=%s samples=%s template=%s thinking=%s",
                dataset_name, dataset_split, max_samples, template_id, enable_thinking)

    process_data.main(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        max_samples=max_samples,
        file_name=file_name,
        template_id=template_id,
        enable_thinking=enable_thinking,
        model_name=model_name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReasoningTrust entrypoint / pipeline runner")
    sub = parser.add_subparsers(dest="command", required=False)

    # toxicity subcommand (default)
    tox = sub.add_parser("toxicity", help="Run toxicity pipeline (default)")
    tox.add_argument("--dataset-name", type=str, default="AI-Secure/DecodingTrust")
    tox.add_argument("--dataset-config", type=str, default="toxicity")
    tox.add_argument("--dataset-split", type=str, default="realtoxicityprompts.toxic")
    tox.add_argument("--max-samples", type=int, default=200)
    tox.add_argument("--file-name", type=str, default="Role_Playing_13_DeepSeek.xlsx")
    tox.add_argument("--template-id", type=int, default=12)
    tox.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tox.add_argument("--enable-thinking", action="store_true", help="Enable thinking extraction")
    tox.add_argument("--results-file", type=str, default=None, help="Path to write results Excel file")  # new

    # allow calling without subcommand: parse same args at top-level
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-config", type=str)
    parser.add_argument("--dataset-split", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--file-name", type=str)
    parser.add_argument("--template-id", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--results-file", type=str)

    return parser


def main(argv: Optional[list] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # prefer explicit subcommand; if none provided, treat as toxicity run
    cmd = args.command or "toxicity"

    # resolve values: prefer subparser values when used, otherwise top-level
    def val(name, default):
        v = getattr(args, name, None)
        return v if v is not None else default

    if cmd == "toxicity":
        run_toxicity(
            dataset_name=val("dataset_name", "AI-Secure/DecodingTrust"),
            dataset_config=val("dataset_config", "toxicity"),
            dataset_split=val("dataset_split", "realtoxicityprompts.toxic"),
            max_samples=val("max_samples", 2),
            file_name=val("file_name", "Role_Playing_13_DeepSeek_NEW.xlsx"),
            template_id=val("template_id", 12),
            model_name=val("model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
        )
    else:
        logger.error("Unknown command: %s", cmd)
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()