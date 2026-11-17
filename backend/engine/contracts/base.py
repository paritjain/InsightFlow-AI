from typing import Any, Dict, Iterator
from pydantic import BaseModel, ConfigDict


class GovernedContractBase(BaseModel):
    """
    Base class for all InsightFlow Governed Contracts.
    Built on Pydantic v2 with seamless dictionary compatibility,
    enabling both attribute access (contract.field) and dict access (contract["field"]).
    """
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__ and key in self.__pydantic_extra__:
            return self.__pydantic_extra__[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.__class__.model_fields:
            setattr(self, key, value)
        else:
            if self.__pydantic_extra__ is None:
                object.__setattr__(self, "__pydantic_extra__", {})
            self.__pydantic_extra__[key] = value

    def __contains__(self, key: str) -> bool:
        if hasattr(self, key) and getattr(self, key, None) is not None:
            return True
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__ and key in self.__pydantic_extra__:
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            val = self[key]
            return val if val is not None else default
        except KeyError:
            return default

    def keys(self):
        k = list(self.model_fields.keys())
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            k.extend(list(self.__pydantic_extra__.keys()))
        return k

    def items(self):
        return [(k, self[k]) for k in self.keys()]
        return [(k, self[k]) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        return self.model_dump(*args, **kwargs)
