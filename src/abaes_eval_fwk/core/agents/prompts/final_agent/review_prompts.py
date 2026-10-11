"""Prompt templates for the Reviewer admissibility checks."""

FINAL_PROMPT_REVIEW_ITEM_SYSTEM_TEMPLATE = """\
Review the assessment shown below against two criteria:
1. Rubric consistency: does the justification support the assigned score and align with the \
rubric guidelines at that score level?
2. Evidence grounding: are the claims made in the justification accurate and supported by \
evidence actually present in the student submission?

Decide, independently, whether the assessment satisfies each criterion. For every criterion you \
mark as not satisfied, state in that criterion's justification field exactly what fails; leave \
that field empty for a criterion you mark as satisfied."""

FINAL_PROMPT_REVIEW_ITEM_USER_TEMPLATE = """\
Rubric guidelines:
\"\"\"{guidelines}\"\"\"

Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Student submission:
\"\"\"{student_work}\"\"\"

The assessment under review:
- Criterion: {item_label}
- Score range: {score_min}–{score_max}
- Score: {score}
- Justification: \"\"\"{justification}\"\"\"

Apply the procedure and JSON output format from the system message to produce your review."""
