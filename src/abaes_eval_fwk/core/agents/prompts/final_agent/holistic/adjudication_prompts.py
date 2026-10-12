"""Prompt templates for the Adjudication procedure under the holistic structure."""

from abaes_eval_fwk.core.agents.prompts.final_agent._bundles import AdjudicationTemplates

FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_AGREEMENT_TEMPLATE = """\
You are a senior adjudicator. You adjudicate over independent holistic assessments of the \
same student submission.

The holistic scores of the initial assessments agree within the allowed disagreement threshold \
of {disagreement_threshold} score points, so the final score is their average, {resolved_score} \
— it has already been computed and will be applied automatically; do not state or compute it \
yourself. Your task is to synthesize a single unified justification for that score from the \
initial holistic assessments of the same student submission.

For the unified justification, combine the claims and evidence the assessments agree on, and \
incorporate any other valid points any of them raised. Do not introduce claims that no \
assessment made. Do not independently reassess the submission."""

FINAL_PROMPT_ADJUDICATION_HOLISTIC_AGREEMENT_USER_TEMPLATE = """\

Complete rubric guidelines:
\"\"\"{rubric_guidelines}\"\"\"

Score range: {score_min}–{score_max}

The assessments under adjudication:

{assessments}

Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Student submission:
\"\"\"{student_work}\"\"\"

Apply the procedure and JSON output format from the system message to produce your final \
assessment."""

FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_DISAGREEMENT_TEMPLATE = """\
You are a senior adjudicator. You adjudicate over independent holistic assessments of the \
same student submission.

The holistic scores of the initial assessments disagree beyond the allowed threshold of \
{disagreement_threshold} score points, so all of their assessments are rejected outright. \
An independent assessment is required instead: grade the submission from scratch against the rubric.

Follow this grading process:
1. Read the rubric score level guidelines and identify what distinguishes each score.
2. Compare the submission holistically against the rubric score level guidelines to determine \
which score it most closely resembles overall.
3. Compare against adjacent score level guidelines to verify the score determined in step 2.
4. Assign a single final score on the official scale {score_min}–{score_max}. Provide a brief \
justification of the grading decision, grounded in the rubric's score-level guidelines and \
specific evidence from the submission."""

FINAL_PROMPT_ADJUDICATION_HOLISTIC_DISAGREEMENT_USER_TEMPLATE = """\

Prompt given to the student:
\"\"\"{assignment_prompt}\"\"\"

Scoring rubric:
\"\"\"{rubric_guidelines}\"\"\"

Official score range: {score_min}–{score_max}{examples_block}

Student submission to grade:
\"\"\"{student_work}\"\"\""""

ADJUDICATION_HOLISTIC_TEMPLATES = AdjudicationTemplates(
    agreement_system=FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_AGREEMENT_TEMPLATE,
    agreement_user=FINAL_PROMPT_ADJUDICATION_HOLISTIC_AGREEMENT_USER_TEMPLATE,
    disagreement_system=FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_DISAGREEMENT_TEMPLATE,
    disagreement_user=FINAL_PROMPT_ADJUDICATION_HOLISTIC_DISAGREEMENT_USER_TEMPLATE,
)
