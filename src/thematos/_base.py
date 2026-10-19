from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Base for thematos config objects: unknown keys are errors, tomotopy/numpy values allowed."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
