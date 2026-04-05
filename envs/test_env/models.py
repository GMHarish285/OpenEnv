from typing import Dict, Optional, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class RagAction(Action):
    """Action for the RagOptimizerEnv."""
    
    action_type: Literal["read_document", "update_document", "delete_document", "add_metadata", "submit"] = Field(
        ..., description="The type of action to perform. 'submit' ends the episode."
    )
    doc_id: Optional[str] = Field(None, description="ID of the document to target.")
    text: Optional[str] = Field(None, description="Text for update_document.")
    metadata_key: Optional[str] = Field(None, description="Metadata key to update.")
    metadata_value: Optional[str] = Field(None, description="Metadata value to set.")


class RagObservation(Observation):
    """Observation from the RagOptimizerEnv."""
    
    message: str = Field(default="", description="Feedback from the environment.")
    current_docs: Dict[str, Dict] = Field(default_factory=dict, description="Summary of current documents in the Knowledge Base (IDs and Metadata).")
