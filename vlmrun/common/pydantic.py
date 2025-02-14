"""Pydantic utilities for VLMRun."""

from datetime import date, datetime, time, timedelta
from typing import Any, List, Type, Union, get_origin

from pydantic import BaseModel, create_model


def patch_response_format(response_format: Type[BaseModel]) -> Type[BaseModel]:
    """Patch the OpenAI response format to handle Pydantic models, including nested models.

    The following fields are not supported by OpenAI:
    - date
    - datetime
    - time
    - timedelta

    This function patches the response format to handle these fields. We convert them to strings and
    then convert them back to the original type.
    """

    def patch_pydantic_field_annotation(annotation: Any) -> Any:
        """Convert unsupported types to string."""
        if annotation in [date, datetime, time, timedelta]:
            return str
        elif get_origin(annotation) is Union:
            # For Union types, convert each argument
            args = annotation.__args__
            if not args:
                return str
            # Convert the first type
            first_type = patch_pydantic_field_annotation(args[0])
            # For single-type unions, combine with str
            if len(args) == 1:
                return Union[first_type, str]
            # For multi-type unions, use first two types
            second_type = patch_pydantic_field_annotation(args[1])
            return Union[first_type, second_type]
        elif get_origin(annotation) is List:
            return List[patch_pydantic_field_annotation(annotation.__args__[0])]
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return patch_pydantic_model(annotation)
        return annotation

    def patch_pydantic_model(model: Type[BaseModel]) -> Type[BaseModel]:
        """Create a new model with patched field types."""
        fields = model.model_fields.copy()
        new_fields = {
            field_name: (patch_pydantic_field_annotation(field.annotation), field)
            for field_name, field in fields.items()
        }
        return create_model(
            f"{model.__name__}_patched", __base__=BaseModel, **new_fields
        )

    return patch_pydantic_model(response_format)
