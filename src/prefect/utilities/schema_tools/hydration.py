import functools
import json
from typing import Any, Callable, Dict, Optional, List, Union

import jinja2
import pydantic
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypeAlias

from prefect.client.schemas import BlockDocument
from prefect.types import StrictVariableValue


class HydrationContext(BaseModel):
    workspace_variables: Dict[
        str,
        StrictVariableValue,
    ] = Field(default_factory=dict)
    workspace_block_documents: List[BlockDocument] = Field(default_factory=list)
    render_workspace_variables: bool = Field(default=False)
    raise_on_error: bool = Field(default=False)
    render_jinja: bool = Field(default=False)
    jinja_context: Dict[str, Any] = Field(default_factory=dict)
    render_workspace_block_documents: bool = Field(default=False)

    @classmethod
    async def build(
        cls,
        session: AsyncSession,
        raise_on_error: bool = False,
        render_jinja: bool = False,
        render_workspace_variables: bool = False,
        render_workspace_block_documents: bool = False,
    ) -> "HydrationContext":
        from prefect.server.models.variables import read_variables
        from prefect.server.models.block_documents import read_block_documents

        if render_workspace_variables:
            variables = await read_variables(
                session=session,
            )
        else:
            variables = []

        if render_workspace_block_documents:
            block_documents = await read_block_documents(
                session=session,
            )
        else:
            block_documents = []

        return cls(
            workspace_variables={
                variable.name: variable.value for variable in variables
            },
            workspace_block_documents=block_documents,
            raise_on_error=raise_on_error,
            render_jinja=render_jinja,
            render_workspace_variables=render_workspace_variables,
            render_workspace_block_documents=render_workspace_block_documents,
        )

    @functools.cached_property
    def workspace_block_documents_by_slug(self) -> Dict[str, Any]:
        return {
            f"{doc.block_type_name}/{doc.name}": doc.data
            for doc in self.workspace_block_documents
        }

    @functools.cached_property
    def workspace_block_documents_by_id(self) -> Dict[str, Any]:
        return {str(doc.id): doc.data for doc in self.workspace_block_documents}


Handler: TypeAlias = Callable[[dict, HydrationContext], Any]
PrefectKind: TypeAlias = Optional[str]

_handlers: Dict[PrefectKind, Handler] = {}


class Placeholder:
    def __eq__(self, other):
        return isinstance(other, type(self))

    @property
    def is_error(self) -> bool:
        return False


class RemoveValue(Placeholder):
    pass


def _remove_value(value) -> bool:
    return isinstance(value, RemoveValue)


class SerializablePlaceholder(pydantic.BaseModel, Placeholder):
    """Serializable Placeholder.

    Used to represent a Placeholder that can be serialized to a JSON,
    especially when the Placeholder needs to be sent over API or saved to a database.

    """


class HydrationError(Placeholder, Exception):
    def __init__(self, detail: Optional[str] = None):
        self.detail = detail

    @property
    def is_error(self) -> bool:
        return True

    @property
    def message(self):
        raise NotImplementedError("Must be implemented by subclass")

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.message == other.message

    def __str__(self):
        return self.message


class KeyNotFound(HydrationError):
    @property
    def message(self):
        return f"Missing '{self.key}' key in __prefect object"

    @property
    def key(self) -> str:
        raise NotImplementedError("Must be implemented by subclass")


class ValueNotFound(KeyNotFound):
    @property
    def key(self):
        return "value"


class TemplateNotFound(KeyNotFound):
    @property
    def key(self):
        return "template"


class VariableNameNotFound(KeyNotFound):
    @property
    def key(self):
        return "variable_name"


class InvalidJSON(HydrationError):
    @property
    def message(self):
        message = "Invalid JSON"
        if self.detail:
            message += f": {self.detail}"
        return message


class InvalidJinja(HydrationError):
    @property
    def message(self):
        message = "Invalid jinja"
        if self.detail:
            message += f": {self.detail}"
        return message


class WorkspaceVariableNotFound(HydrationError):
    @property
    def variable_name(self) -> str:
        assert self.detail is not None
        return self.detail

    @property
    def message(self):
        return f"Variable '{self.detail}' not found in workspace."


class WorkspaceVariable(SerializablePlaceholder):
    variable_name: str

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and self.variable_name == other.variable_name
        )

    @pydantic.model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            "__prefect_kind": "workspace_variable",
            "variable_name": self.variable_name,
        }


class WorkspaceBlockDocument(SerializablePlaceholder):
    reference: dict[str, str]

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.reference == other.reference

    @pydantic.model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            "__prefect_kind": "workspace_block_document",
            **self.reference,
        }

    @property
    def block_document_id(self) -> Optional[str]:
        return self.reference.get("block_document_id", None)

    @property
    def block_document_slug(self) -> Optional[str]:
        return self.reference.get("block_document_slug", None)


class WorkspaceBlockDocumentNotFound(HydrationError):
    @property
    def message(self):
        return f"Block document '{self.detail}' not found in workspace."


class ValidJinja(SerializablePlaceholder):
    template: str

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.template == other.template

    @pydantic.model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            "__prefect_kind": getattr(self, "__prefect_kind"),
            "template": self.template,
        }


def handler(kind: PrefectKind) -> Callable:
    def decorator(func: Handler) -> Handler:
        _handlers[kind] = func
        return func

    return decorator


def call_handler(kind: PrefectKind, obj: dict, ctx: HydrationContext) -> Any:
    if kind not in _handlers:
        return (obj or {}).get("value", None)

    res = _handlers[kind](obj, ctx)
    if ctx.raise_on_error and isinstance(res, HydrationError):
        raise res
    return res


@handler("none")
def null_handler(obj: dict, ctx: HydrationContext):
    if "value" in obj:
        # null handler is a pass through, so we want to continue to hydrate
        return _hydrate(obj["value"], ctx)
    else:
        return ValueNotFound()


@handler("json")
def json_handler(obj: dict, ctx: HydrationContext):
    if "value" in obj:
        if isinstance(obj["value"], dict):
            dehydrated_json = _hydrate(obj["value"], ctx)
        else:
            dehydrated_json = obj["value"]

        # If the result is a Placeholder, we should return it as is
        if isinstance(dehydrated_json, Placeholder):
            return dehydrated_json

        try:
            return json.loads(dehydrated_json)
        except (json.decoder.JSONDecodeError, TypeError) as e:
            return InvalidJSON(detail=str(e))
    else:
        # If `value` is not in the object, we need special handling to help
        # the UI. For now if an object looks like {"__prefect_kind": "json"}
        # We will remove it from the parent object. e.x.
        # {"a": {"__prefect_kind": "json"}} -> {}
        # or
        # [{"__prefect_kind": "json"}] -> []
        return RemoveValue()


@handler("jinja")
def jinja_handler(obj: dict, ctx: HydrationContext):
    from prefect.server.utilities.user_templates import (
        TemplateSecurityError,
        render_user_template_sync,
        validate_user_template,
    )

    if "template" in obj:
        if isinstance(obj["template"], dict):
            dehydrated_jinja = _hydrate(obj["template"], ctx)
        else:
            dehydrated_jinja = obj["template"]

        # If the result is a Placeholder, we should return it as is
        if isinstance(dehydrated_jinja, Placeholder):
            return dehydrated_jinja

        try:
            validate_user_template(dehydrated_jinja)
        except (jinja2.exceptions.TemplateSyntaxError, TemplateSecurityError) as exc:
            return InvalidJinja(detail=str(exc))

        if ctx.render_jinja:
            jinja_context = ctx.jinja_context.copy()
            if ctx.render_workspace_block_documents:
                blocks = {}
                for block in ctx.workspace_block_documents:
                    if block.block_type_name not in blocks:
                        blocks[block.block_type_name] = {}
                    blocks[block.block_type_name][block.name] = block.data

                jinja_context["prefect"] = {
                    "blocks": blocks,
                }
            if ctx.render_workspace_variables:
                jinja_context["prefect"]["variables"] = ctx.workspace_variables
            return render_user_template_sync(dehydrated_jinja, ctx.jinja_context)
        else:
            return ValidJinja(template=dehydrated_jinja)
    else:
        return TemplateNotFound()


@handler("workspace_variable")
def workspace_variable_handler(obj: dict, ctx: HydrationContext):
    if "variable_name" in obj:
        if isinstance(obj["variable_name"], dict):
            dehydrated_variable = _hydrate(obj["variable_name"], ctx)
        else:
            dehydrated_variable = obj["variable_name"]

        # If the result is a Placeholder, we should return it as is
        if isinstance(dehydrated_variable, Placeholder):
            return dehydrated_variable

        if not ctx.render_workspace_variables:
            return WorkspaceVariable(variable_name=obj["variable_name"])

        if dehydrated_variable in ctx.workspace_variables:
            return ctx.workspace_variables[dehydrated_variable]
        else:
            return WorkspaceVariableNotFound(detail=dehydrated_variable)
    else:
        # Special handling if `variable_name` is not in the object.
        # If an object looks like {"__prefect_kind": "workspace_variable"}
        # we will remove it from the parent object. e.x.
        # {"a": {"__prefect_kind": "workspace_variable"}} -> {}
        # or
        # [{"__prefect_kind": "workspace_variable"}] -> []
        # or
        # {"__prefect_kind": "workspace_variable"} -> {}
        return RemoveValue()


@handler("workspace_block_document")
def workspace_block_document_handler(obj: dict, ctx: HydrationContext):
    reference_types = ["block_document_slug", "block_document_id"]

    if any(ref_type in obj for ref_type in reference_types):
        if "block_document_slug" in obj:
            reference = obj["block_document_slug"]
            block_documents = ctx.workspace_block_documents_by_slug
        else:
            reference = obj["block_document_id"]
            block_documents = ctx.workspace_block_documents_by_id

        if isinstance(reference, dict):
            dehydrated_block_document = _hydrate(reference, ctx)
        else:
            dehydrated_block_document = reference

        # If the result is a Placeholder, we should return it as is
        if isinstance(dehydrated_block_document, Placeholder):
            return dehydrated_block_document

        if not ctx.render_workspace_block_documents:
            return WorkspaceBlockDocument(reference=obj)

        if dehydrated_block_document in block_documents:
            return block_documents[dehydrated_block_document]

        return WorkspaceBlockDocumentNotFound(detail=dehydrated_block_document)

    else:
        # Special handling if none of reference types are in the object.
        # If an object looks like {"__prefect_kind": "workspace_reference"}
        # we will remove it from the parent object. e.x.
        # {"a": {"__prefect_kind": "workspace_reference"}} -> {}
        # or
        # [{"__prefect_kind": "workspace_reference"}] -> []
        # or
        # {"__prefect_kind": "workspace_reference"} -> {}
        return RemoveValue()


def hydrate(obj: dict, ctx: Optional[HydrationContext] = None):
    res = _hydrate(obj, ctx)

    if _remove_value(res):
        return {}

    return res


def _hydrate(obj, ctx: Optional[HydrationContext] = None) -> Any:
    if ctx is None:
        ctx = HydrationContext()

    prefect_object = isinstance(obj, dict) and "__prefect_kind" in obj
    reference_object = isinstance(obj, dict) and "$ref" in obj

    if prefect_object:
        prefect_kind = obj.get("__prefect_kind")
        return call_handler(prefect_kind, obj, ctx)
    elif reference_object:  # This handles references by $ref format
        if "workspace_variable" in obj["$ref"]:
            return call_handler("workspace_variable", obj["$ref"], ctx)
        return call_handler(
            "workspace_block_document",
            obj["$ref"]
            if isinstance(obj["$ref"], dict)
            else {"block_document_id": obj["$ref"]},
            ctx,
        )
    else:
        if isinstance(obj, dict):
            return {
                key: hydrated_value
                for key, value in obj.items()
                if not _remove_value(hydrated_value := _hydrate(value, ctx))
            }
        elif isinstance(obj, list):
            return [
                hydrated_element
                for element in obj
                if not _remove_value(hydrated_element := _hydrate(element, ctx))
            ]
        else:
            return obj


def collect_placeholders(obj: Union[Dict[str, Any], Placeholder]) -> List[Placeholder]:
    """
    Recursively collects all placeholder values embedded within the parameters
    dictionary, including templates and workspace variables
    """
    placeholders = []

    if isinstance(obj, Placeholder):
        return [obj]

    for _, value in obj.items():
        if isinstance(value, dict):
            placeholders += collect_placeholders(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    placeholders += collect_placeholders(item)
                elif isinstance(item, Placeholder):
                    placeholders.append(item)
        elif isinstance(value, Placeholder):
            placeholders.append(value)
    return placeholders
