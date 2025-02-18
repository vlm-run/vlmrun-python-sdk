from typing import Type, Dict, Any
from pydantic import BaseModel, create_model
from graphql import parse
from graphql.language.ast import (
    SelectionSetNode,
    FieldNode,
    OperationDefinitionNode,
    DocumentNode,
)
from loguru import logger


def create_pydantic_model_from_gql(
    base_model: Type[BaseModel], gql_query: str
) -> Type[BaseModel]:
    """Creates a subset Pydantic model based on a GraphQL query.

    Args:
        base_model: The original Pydantic model containing all fields
        gql_query: GraphQL query string specifying desired fields

    Returns:
        A new Pydantic model containing only the fields specified in the query

    Raises:
        graphql.GraphQLSyntaxError: If the GQL query is invalid
        ValueError: If the query references fields not in the base model
    """
    try:
        ast: DocumentNode = parse(gql_query)
    except Exception as e:
        logger.error(f"Invalid GraphQL query: {e}")
        raise

    fields: Dict[str, Any] = {}

    def extract_fields(
        node: SelectionSetNode,
        current_model: Type[BaseModel],
        current_fields: Dict[str, Any],
    ) -> None:
        if isinstance(node, SelectionSetNode):
            for selection in node.selections:
                if isinstance(selection, FieldNode):
                    field_name = selection.name.value
                    original_field = current_model.model_fields.get(field_name)

                    if original_field is None:
                        if current_model == base_model:
                            raise ValueError(f"Field '{field_name}' not found in model")
                        else:
                            raise ValueError(
                                f"Field '{field_name}' not found in nested model"
                            )

                    if selection.selection_set:
                        if not hasattr(original_field.annotation, "model_fields"):
                            raise ValueError(
                                f"Cannot query nested fields of scalar type: {field_name}"
                            )

                        nested_fields = {}
                        extract_fields(
                            selection.selection_set,
                            original_field.annotation,
                            nested_fields,
                        )
                        if nested_fields:
                            current_fields[field_name] = (
                                create_model(
                                    f"{current_model.__name__}{field_name.capitalize()}",
                                    **nested_fields,
                                ),
                                original_field.default,
                            )
                    else:
                        current_fields[field_name] = (
                            original_field.annotation,
                            original_field.default,
                        )

    for definition in ast.definitions:
        if isinstance(definition, OperationDefinitionNode):
            extract_fields(definition.selection_set, base_model, fields)

    if not fields:
        raise ValueError("No valid fields found in GraphQL query")

    return create_model(f"{base_model.__name__}", __base__=BaseModel, **fields)
