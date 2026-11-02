from fire import Fire
from sys import stderr

from .cli import RagCLI


def main() -> None:
    """Run the command-line interface."""
    try:
        Fire(RagCLI)
    except Exception as exc:
        print(f"Error: {exc}", file=stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nProgram interrupted!\n")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
