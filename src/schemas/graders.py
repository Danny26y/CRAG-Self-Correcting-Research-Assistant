from typing import Literal
from pydantic import BaseModel, Field

class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if the document is relevant to user query 'no' if it's not."

    )
    explanation: str = Field(
        description="Brief 1 sentence explanation of why the document is marked a relevant or irrelevant."
    )

class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Grounding Score: 'yes' if every claim in the answer is backed by facts in the context, 'no' if not."
    )
    explanation: str = Field(
        description="Brief explanation of any underground claims found"
    )

class GradeAnswering(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Completeness score: 'yes' if the answer directly addresses the user query, 'no' if not."
    )
    explanation: str = Field(
        description="Brief explanation of what was missing in the user query."
    )
