# When the statistical score is shown

The classifier learned from three large repositories: microsoft/vscode,
facebook/react and tensorflow/tensorflow, 2024 issues only. Asked about any
other repository it still returns a percentage, computed from author metadata
and text patterns that carry different information outside the population it
was fitted on.

That is not hypothetical. GHIC scored a real, reproducible bug at 26% —
"Likely not actionable" — on a small personal repository. The bug was fixed
within the hour. The number was not merely wrong; it was presented as a
finding, next to an LLM review that had diagnosed the problem correctly.

## The rule

`ServiceSettings.score_is_calibrated_for(repo)` is true when the repository is
either listed in `GHIC_CALIBRATED_REPOS` or has an entry in
`GHIC_REPO_THRESHOLDS`. A per-repo threshold counts on its own, because one
only exists after `python -m ghic.backtest` measured that repository's
held-out issues and the tuned value beat the global default.

Both are empty by default. A deployment claims calibration explicitly; the
product does not assume it.

When a repository is not calibrated:

- the headline verdict becomes **Needs maintainer review** rather than
  "Likely actionable" / "Likely not actionable". The verdict is the model's
  claim, so withholding the score while keeping the verdict would be
  incoherent.
- the statistical score row states why it is absent instead of showing the
  number with a caveat. A percentage with a footnote still anchors the reader
  on the percentage.
- the disagreement banner is suppressed. It says two signals conflict; with no
  score worth trusting there is nothing to conflict with.
- **the LLM review is untouched.** It is what the maintainer opened the comment
  for, and it does not depend on the training distribution.

This reuses the existing "Needs maintainer review" vocabulary rather than
inventing a second way of saying the same thing.

## Offline and served do not agree, and cannot

Measured on the same test slice at the same threshold (`scripts/parity_check.py`):

| repository | n | F1 offline | F1 served | recall offline | recall served | AUC offline | AUC served |
|---|---|---|---|---|---|---|---|
| facebook/react | 129 | 0.750 | 0.750 | 0.783 | 0.783 | 0.894 | 0.895 |
| microsoft/vscode | 820 | 0.411 | 0.302 | 0.288 | 0.192 | 0.833 | 0.841 |
| tensorflow/tensorflow | 228 | 0.850 | 0.850 | 0.850 | 0.850 | 0.929 | 0.929 |
| overall | 1177 | 0.668 | 0.637 | 0.575 | 0.529 | 0.881 | 0.885 |

Two repositories match to three decimals. vscode's recall drops while its AUC
does not — it is slightly *higher* served. Equal ranking with a different
operating point is a threshold effect, not features arriving corrupted.

The cause is `author_is_first_time_contributor`, computed chronologically
within the dataframe it is given. Offline that dataframe is the whole slice and
76.3% of issues are flagged first-time. The webhook has one issue, cannot see
the author's prior history, and flags 100%. Roughly a quarter of issues
therefore reach the model with a different value than they had in training,
which shifts probabilities down slightly and moves borderline issues below a
fixed 0.5 cut. vscode has the most issues near that boundary, so it moves most.

This is an information asymmetry, not a bug, and it cannot be closed by making
the two numbers equal. The options are to fetch the author's issue history at
inference (an extra GitHub call per issue, unproven value), to drop the feature
and retrain (retraining is deliberately disabled), or to accept the shift and
choose the operating point on served probabilities.

GHIC does the third. This is why `ghic.backtest` replays through the real
webhook rather than scoring offline: the thresholds it recommends are tuned on
the probabilities production actually produces. **Do not take a threshold from
`reports/metrics.json`** — those are offline numbers and describe a model that
sees more than the deployed one does.

## Configuring it

Run the backtest, then set what it recommends:

```
python -m ghic.backtest
GHIC_REPO_THRESHOLDS=microsoft/vscode=0.18
GHIC_CALIBRATED_REPOS=facebook/react,tensorflow/tensorflow
```

vscode needs no entry in the second variable — its threshold implies it.
react and tensorflow need no threshold: their tuned values did not beat 0.5 on
held-out data, and the backtest refuses to recommend a threshold that loses.
