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
    except KeyboardInterrupt:
        print("\nProgram interrupted!\n")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
