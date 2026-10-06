from .models import MinimalSource


def source_matches(
    retrieved: MinimalSource,
    expected: MinimalSource,
    threshold: float = 0.05,
) -> bool:
    """Check whether two source locations overlap sufficiently."""
    if retrieved.file_path != expected.file_path:
        return False

    intersection = max(
        0,
        min(retrieved.last_character_index,
            expected.last_character_index)
        - max(retrieved.first_character_index,
              expected.first_character_index),
    )

    retrieved_length = (
        retrieved.last_character_index
        - retrieved.first_character_index
    )
    expected_length = (
        expected.last_character_index
        - expected.first_character_index
    )
    union = retrieved_length + expected_length - intersection

    return union > 0 and intersection / union >= threshold


def recall_for_question(
    retrieved: list[MinimalSource],
    expected: list[MinimalSource],
    k: int,
) -> float:

    if not expected:
        raise ValueError("Cannot calculate recall without expected sources.")

    if type(k) is not int or k < 0:
        raise ValueError("k should be a positive integer!")

    found = 0

    for target in expected:
        for result in retrieved[:k]:
            if source_matches(result, target):
                found += 1
                break
    recall = found / len(expected)
    return recall
