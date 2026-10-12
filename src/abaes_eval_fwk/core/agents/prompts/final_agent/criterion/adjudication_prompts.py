"""Prompt templates for the Adjudication procedure under the criterion structure."""

from abaes_eval_fwk.core.agents.prompts.final_agent._bundles import AdjudicationTemplates

FINAL_PROMPT_ADJUDICATION_CRITERION_INTRO_SYSTEM_TEMPLATE = """\
You are a senior adjudicator. You adjudicate over the initial criterion-level assessments of \
this submission's rubric criteria, one criterion at a time. For each criterion, if the \
assessment scores agree within the allowed threshold you will synthesize a single unified \
justification from their assessments (their average score is precomputed and applied \
automatically); if they disagree those assessments are set aside and your own independent \
assessment of that criterion — its score and justification — is adopted instead, unchanged. \
Once every criterion has been adjudicated, you will synthesize one justification across all of \
them."""

FINAL_PROMPT_ADJUDICATION_CRITERION_INTRO_USER_TEMPLATE = """\
Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Student submission:
\"\"\"{student_work}\"\"\""""

FINAL_PROMPT_ADJUDICATION_CRITERION_ITEM_SYSTEM_AGREEMENT_TEMPLATE = """\
This criterion's scores agree within the allowed disagreement threshold of \
{disagreement_threshold} score points, so the final score is their average, {resolved_score} — it \
has already been computed and will be applied automatically; do not state or compute it \
yourself. Synthesize a single unified justification for that score from the initial assessments \
of this criterion.

For the unified justification, combine the claims and evidence the assessments agree on, and \
incorporate any other valid points any of them raised. Do not introduce claims that no \
assessment made. Do not independently reassess the submission.

Adjudicate only this criterion — ignore claims and evidence made by assessments of other \
criteria."""

FINAL_PROMPT_ADJUDICATION_CRITERION_AGREEMENT_USER_TEMPLATE = """\

Criterion: {criterion_name}

Rubric guidelines for this criterion:
\"\"\"{rubric_guidelines}\"\"\"

Score range: {score_min}–{score_max}

The assessments under adjudication:

{assessments}

Apply the procedure and JSON output format from the system message to produce your adjudication \
for this criterion."""

FINAL_PROMPT_ADJUDICATION_CRITERION_ITEM_SYSTEM_DISAGREEMENT_TEMPLATE = """\
This criterion's scores disagree beyond the allowed threshold of {disagreement_threshold} score \
points, so all of their assessments are rejected outright. An independent assessment of this \
criterion is required instead: grade it from scratch against its guidelines.

Follow this grading process:
1. Read the criterion score level guidelines below and identify what distinguishes each score.
2. Compare the submission against the criterion score level guidelines to determine which score \
it most closely resembles for this criterion. Evaluate only this criterion — ignore strengths or \
weaknesses in other dimensions.
3. Compare against adjacent score level guidelines to verify the score determined in step 2.
4. Assign a score within {score_min}–{score_max}. Provide a brief justification of your grading \
decision, grounded in the criterion's score-level guidelines and specific evidence from the \
submission."""

FINAL_PROMPT_ADJUDICATION_CRITERION_DISAGREEMENT_USER_TEMPLATE = """\

Criterion: {criterion_name}

Rubric guidelines for this criterion:
\"\"\"{rubric_guidelines}\"\"\"

Official score range for this criterion: {score_min}–{score_max}{examples_block}"""

FINAL_PROMPT_ADJUDICATION_CRITERION_FINAL_SYSTEM_TEMPLATE = """\
Every criterion has now been adjudicated above. The overall score has already been computed \
from their combined final scores and applied automatically; do not state or compute it \
yourself.

For the justification, combine the claims and evidence the per-criterion justifications raise, \
connecting them to the rubric's score-level guidelines to explain how they support the overall \
score. Do not introduce claims that no assessment made."""

FINAL_PROMPT_ADJUDICATION_CRITERION_FINAL_USER_TEMPLATE = """\
Per-criterion final outcomes:

{criteria_summary}

Overall score: {overall_score}

Apply the procedure and JSON output format from the system message to produce your synthesis \
across all criteria."""

ADJUDICATION_CRITERION_TEMPLATES = AdjudicationTemplates(
    intro_system=FINAL_PROMPT_ADJUDICATION_CRITERION_INTRO_SYSTEM_TEMPLATE,
    intro_user=FINAL_PROMPT_ADJUDICATION_CRITERION_INTRO_USER_TEMPLATE,
    intro_fields=("assignment_prompt", "student_work"),
    agreement_system=FINAL_PROMPT_ADJUDICATION_CRITERION_ITEM_SYSTEM_AGREEMENT_TEMPLATE,
    agreement_user=FINAL_PROMPT_ADJUDICATION_CRITERION_AGREEMENT_USER_TEMPLATE,
    disagreement_system=FINAL_PROMPT_ADJUDICATION_CRITERION_ITEM_SYSTEM_DISAGREEMENT_TEMPLATE,
    disagreement_user=FINAL_PROMPT_ADJUDICATION_CRITERION_DISAGREEMENT_USER_TEMPLATE,
    final_system=FINAL_PROMPT_ADJUDICATION_CRITERION_FINAL_SYSTEM_TEMPLATE,
    final_user=FINAL_PROMPT_ADJUDICATION_CRITERION_FINAL_USER_TEMPLATE,
)
