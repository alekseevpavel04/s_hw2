# Based on seminar materials

# Don't forget to support cases when target_text == ''


import numpy as np


def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Character Error Rate (CER) between target and predicted text.
    CER = (edits / total_characters)
    """
    target_text = target_text.strip()
    predicted_text = predicted_text.strip()

    # Compute edit distance
    edit_distance = levenshtein_distance(target_text, predicted_text)
    total_characters = len(target_text)

    if total_characters == 0:  # Handle edge case
        return 0.0 if len(predicted_text) == 0 else 1.0

    return edit_distance / total_characters


def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Word Error Rate (WER) between target and predicted text.
    WER = (edits / total_words)
    """
    target_words = target_text.strip().split()
    predicted_words = predicted_text.strip().split()

    # Compute edit distance
    edit_distance = levenshtein_distance(target_words, predicted_words)
    total_words = len(target_words)

    if total_words == 0:  # Handle edge case
        return 0.0 if len(predicted_words) == 0 else 1.0

    return edit_distance / total_words


def levenshtein_distance(seq1, seq2) -> int:
    """
    Calculate the Levenshtein distance between two sequences.
    This supports both character and word sequences.
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[n][m]
