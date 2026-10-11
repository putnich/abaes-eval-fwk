"""Prompt templates for the Adjudication with Review procedure under the holistic structure."""

from abaes_eval_fwk.core.agents.prompts.final_agent._bundles import (
    AdjudicationWithReviewTemplates,
)

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_INTRO_SYSTEM_TEMPLATE = """\
You are a senior adjudicator over independent holistic assessments of the same student \
submission, each produced by a different grading model. You will first review every assessment \
for admissibility, then adjudicate a final decision from the review outcome. Where the review \
finds any assessment inadmissible, your own independent assessment of this submission — made \
against the rubric without sight of theirs — takes its place in the decision."""

# No rubric block here: each review turn carries the whole-rubric guidelines itself, so stating
# them up front would only repeat them.
FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_INTRO_USER_TEMPLATE = """\
Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Student submission:
\"\"\"{student_work}\"\"\""""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_POSITIVE_SYSTEM_TEMPLATE = """\
Your review found every assessment admissible, so the final score is their average, \
{resolved_score} — it has already been computed and will be applied automatically; do not state \
or compute it yourself.

For the unified justification, combine the claims and evidence the assessments agree on, and \
incorporate any other valid points any of them raised. Do not introduce claims that no \
assessment made. Do not independently reassess the submission."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_POSITIVE_USER_TEMPLATE = """\
The admissible assessments:

{assessments}

Apply the procedure and JSON output format from the system message to produce your final \
decision."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_MIXED_SYSTEM_TEMPLATE = """\
Your review found some assessments admissible and the rest inadmissible. The inadmissible ones \
have been excluded — disregard them entirely, and do not carry over their claims. The admissible \
ones are retained, and your own independent assessment of this submission, made against the \
rubric without sight of any of them, is added to the pool. The final score is the pool's \
average, {resolved_score} — it has already been computed and will be applied automatically; do \
not state or recompute it.

For the unified justification, combine the claims and evidence the retained assessments agree \
on, incorporate any other valid points any of them raised, and state why your own score is \
supported by the rubric guidelines and the submission evidence."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_MIXED_USER_TEMPLATE = """\
The retained admissible assessments and your own independent assessment:

{assessments}

Apply the procedure and JSON output format from the system message to produce your final \
decision."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_NEGATIVE_SYSTEM_TEMPLATE = """\
Your review found no assessment admissible, so all of their assessments are rejected outright. \
An independent assessment is required instead: grade the submission from scratch against the rubric.

Follow this grading process:
1. Read the rubric score level guidelines and identify what distinguishes each score.
2. Compare the submission holistically against the rubric score level guidelines to determine \
which score it most closely resembles overall.
3. Compare against adjacent score level guidelines to verify the score determined in step 2.
4. Assign a single final score on the official scale {score_min}–{score_max}. Provide a brief \
justification of the grading decision, grounded in the rubric's score-level guidelines and \
specific evidence from the submission."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_NEGATIVE_USER_TEMPLATE = """\

Scoring rubric:
\"\"\"{rubric_guidelines}\"\"\"

Official score range: {score_min}–{score_max}{examples_block}"""

ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES = AdjudicationWithReviewTemplates(
    intro_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_INTRO_SYSTEM_TEMPLATE,
    intro_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_INTRO_USER_TEMPLATE,
    intro_fields=("assignment_prompt", "student_work"),
    positive_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_POSITIVE_SYSTEM_TEMPLATE,
    positive_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_POSITIVE_USER_TEMPLATE,
    mixed_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_MIXED_SYSTEM_TEMPLATE,
    mixed_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_MIXED_USER_TEMPLATE,
    negative_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_NEGATIVE_SYSTEM_TEMPLATE,
    negative_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_HOLISTIC_NEGATIVE_USER_TEMPLATE,
)
