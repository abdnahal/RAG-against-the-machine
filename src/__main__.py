import fire
import sys

from .cli import RagCLI


def main() -> None:
    """Run the command-line interface."""
    try:
        fire.Fire(RagCLI)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
