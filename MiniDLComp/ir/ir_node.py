from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IRNode:
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    value: Optional[Any] = None
    users: List[str] = field(default_factory=list)
    deleted: bool = False

    def is_constant(self) -> bool:
        return self.op_type == "const"

    def __repr__(self) -> str:
        return (
            f"IRNode(name={self.name}, op_type={self.op_type}, "
            f"inputs={self.inputs}, attrs={self.attrs}, deleted={self.deleted})"
        )
