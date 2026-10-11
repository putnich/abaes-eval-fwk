"""Prompt templates for the Adjudication with Review procedure under the criterion structure."""

from abaes_eval_fwk.core.agents.prompts.final_agent._bundles import (
    AdjudicationWithReviewTemplates,
)

# No rubric block in the intro — criterion guidelines are shown per criterion, not up front — and
# none in the item turns below either: each criterion's own review turn(s), just above, already
# stated its guidelines, so restating them here would only repeat them.
FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_INTRO_SYSTEM_TEMPLATE = """\
You are a senior adjudicator over independent criterion-specialized assessments of the same \
student submission, each produced by a different grading model. You will adjudicate the rubric's \
criteria one criterion at a time: for each, you will first review every assessment of that \
criterion for admissibility, then adjudicate a final decision for it from the review outcome. \
Where the review finds any assessment inadmissible, your own independent assessment of that \
criterion — made against its guidelines without sight of theirs — takes its place in the \
decision. Once every criterion has been adjudicated, you will synthesize one justification \
across all of them."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_INTRO_USER_TEMPLATE = """\
Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Student submission:
\"\"\"{student_work}\"\"\""""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_POSITIVE_SYSTEM_TEMPLATE = """\
Your review found every assessment of this criterion admissible, so its final score is their \
average, {resolved_score} — it has already been computed and will be applied automatically; do \
not state or compute it yourself.

For the unified justification, combine the claims and evidence the assessments agree on, and \
incorporate any other valid points any of them raised. Do not introduce claims that no \
assessment made. Do not independently reassess the submission."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_POSITIVE_USER_TEMPLATE = """\
Criterion: {criterion_name}

The admissible assessments:

{assessments}

Apply the procedure and JSON output format from the system message to produce your final \
decision for this criterion."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_MIXED_SYSTEM_TEMPLATE = """\
Your review found some assessments of this criterion admissible and the rest inadmissible. The \
inadmissible ones have been excluded — disregard them entirely, and do not carry over their \
claims. The admissible ones are retained, and your own independent assessment of this criterion, \
made against its guidelines without sight of any of them, is added to the pool. This \
criterion's final score is the pool's average, {resolved_score} — it has already been computed \
and will be applied automatically; do not state or recompute it.

For the unified justification, combine the claims and evidence the retained assessments agree \
on, incorporate any other valid points any of them raised, and state why your own score is \
supported by this criterion's guidelines and the submission evidence."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_MIXED_USER_TEMPLATE = """\
Criterion: {criterion_name}

The retained admissible assessments and your own:

{assessments}

Apply the procedure and JSON output format from the system message to produce your final \
decision for this criterion."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_NEGATIVE_SYSTEM_TEMPLATE = """\
Your review found no assessment of this criterion admissible, so all of their assessments are \
rejected outright. An independent assessment of this criterion is required instead: grade it \
from scratch against its guidelines.

Follow this grading process:
1. Read the criterion score level guidelines below and identify what distinguishes each score.
2. Compare the submission against the criterion score level guidelines to determine which score \
it most closely resembles for this criterion. Evaluate only this criterion — ignore strengths or \
weaknesses in other dimensions.
3. Compare against adjacent score level guidelines to verify the score determined in step 2.
4. Assign a score within {score_min}–{score_max}. Provide a brief justification of your grading \
decision, grounded in the criterion's score-level guidelines and specific evidence from the \
submission."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_NEGATIVE_USER_TEMPLATE = """\

Criterion: {criterion_name}

Rubric guidelines for this criterion:
\"\"\"{rubric_guidelines}\"\"\"

Official score range for this criterion: {score_min}–{score_max}{examples_block}"""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_FINAL_SYSTEM_TEMPLATE = """\
Every criterion has now been adjudicated above, and the overall score has already been computed \
from their combined final scores — it will be applied automatically; do not state or compute it \
yourself.

For the justification, combine the claims and evidence the per-criterion justifications raise, \
connecting them to the rubric's score-level guidelines to explain how they support the overall \
score. Do not introduce claims that no assessment made."""

FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_FINAL_USER_TEMPLATE = """\
Apply the procedure and JSON output format from the system message to produce your synthesis \
across all criteria."""

ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES = AdjudicationWithReviewTemplates(
    intro_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_INTRO_SYSTEM_TEMPLATE,
    intro_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_INTRO_USER_TEMPLATE,
    intro_fields=("assignment_prompt", "student_work"),
    positive_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_POSITIVE_SYSTEM_TEMPLATE,
    positive_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_POSITIVE_USER_TEMPLATE,
    mixed_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_MIXED_SYSTEM_TEMPLATE,
    mixed_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_MIXED_USER_TEMPLATE,
    negative_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_NEGATIVE_SYSTEM_TEMPLATE,
    negative_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_ITEM_NEGATIVE_USER_TEMPLATE,
    final_system=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_FINAL_SYSTEM_TEMPLATE,
    final_user=FINAL_PROMPT_ADJUDICATION_WITH_REVIEW_CRITERION_FINAL_USER_TEMPLATE,
)
