"""Shared redacted runtime failure diagnostics."""

from __future__ import annotations

import re

from agent_hub.models.capacity import (
    CapacityBackendError,
    CapacityConfigurationError,
    CapacityQueueFull,
    CapacityUnavailable,
    CapacityWaitTimeout,
)
from agent_hub.models.gateway import ModelGatewayError
from agent_hub.models.litellm_client import ModelResponseError, ModelTransportError
from agent_hub.models.registry import NoCapableDeployment

MAX_FAILURE_REASON_LENGTH = 240
RuntimeFailureDiagnostic = dict[str, str | int | bool]
SENSITIVE_FAILURE_REASON = re.compile(
    r"(api[_-]?key|authorization|bearer|credential|password|secret|(?:access|refresh|session)[_-]?token|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)
_STATUS_CODE = re.compile(r"\(status=(?P<status>[1-5][0-9]{2})\)")
_CREWAI_STEP_TIMEOUT = re.compile(
    r"^CrewAI step timed out: step=(?P<step>[A-Za-z0-9_.-]{1,128}) actor=(?P<actor>[A-Za-z0-9_.-]{1,128})$"
)
_CAPACITY_CONTEXT = re.compile(
    r"model capacity unavailable "
    r"\(logical_models=(?P<logical_models>[A-Za-z0-9_,:-]{1,512}); "
    r"deployments=(?P<deployments>[A-Za-z0-9_,:-]{1,1024})\)"
)
_MODEL_CONTEXT = re.compile(
    r"\((?:status=(?P<status>[1-5][0-9]{2}); )?"
    r"logical_models=(?P<logical_models>[A-Za-z0-9_,:-]{1,512}); "
    r"deployments=(?P<deployments>[A-Za-z0-9_,:-]{1,1024})\)"
)
_MODEL_REQUEST_CHECKPOINT_MISMATCH = re.compile(
    r"^model request changed after checkpoint "
    r"\(step=(?P<step>[A-Za-z0-9_.-]{1,128}); "
    r"actor=(?P<actor>[A-Za-z0-9_.-]{1,128}); "
    r"purpose=(?P<purpose>[A-Za-z0-9_.-]{1,64}); "
    r"call_index=(?P<call_index>[0-9]{1,6}); "
    r"expected=(?P<expected>[A-Fa-f0-9]{6,64}); "
    r"actual=(?P<actual>[A-Fa-f0-9]{6,64})\)$"
)
_SAFE_MODEL_CONTEXT_VALUE = re.compile(r"^[A-Za-z0-9_,:-]{1,1024}$")
GENERIC_MODEL_GATEWAY_FAILURE = "model gateway failed"
LEGACY_GENERIC_FAILURES = frozenset(
    {
        "model gateway failed",
        "discussion_failed",
        "dispatch execution failed",
        "step execution failed",
        "runtime_failed",
    }
)
SAFE_MODEL_GATEWAY_FAILURES = frozenset(
    {
        "model capacity unavailable",
        "model transport failed",
        "model outcome recording failed",
        "model capacity release failed",
        "model gateway completed without a response",
        "model response text is empty",
        "model response is empty",
    }
)


def safe_model_gateway_failure_reason(error: Exception) -> str | None:
    """Return a stable model-gateway diagnostic that is safe to show in the UI."""

    if isinstance(error, ModelTransportError):
        base = (
            "model response failed"
            if isinstance(error, ModelResponseError)
            else "model transport failed"
        )
        suffix = _model_context_suffix(error, status_code=error.status_code)
        if suffix:
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {base} {suffix}"
        if error.status_code is not None:
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {base} (status={error.status_code})"
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {base}"
    if isinstance(error, NoCapableDeployment):
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: no capable deployment"
    if isinstance(error, CapacityQueueFull):
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity queue is full"
    if isinstance(error, CapacityWaitTimeout):
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity queue timeout"
    if isinstance(error, CapacityUnavailable):
        message = normalize_failure_reason(str(error))
        if (
            _CAPACITY_CONTEXT.fullmatch(message) is not None
            and is_safe_failure_reason(message)
        ):
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {message}"
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity unavailable"
    if isinstance(error, CapacityConfigurationError):
        suffix = _model_context_suffix(error)
        if suffix:
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity configuration failed {suffix}"
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity configuration failed"
    if isinstance(error, CapacityBackendError):
        suffix = _model_context_suffix(error)
        if suffix:
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity backend failed {suffix}"
        return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model capacity backend failed"
    if isinstance(error, ModelGatewayError):
        message = normalize_failure_reason(str(error))
        suffix = _model_context_suffix(error)
        if message == "model credential resolution failed":
            if suffix:
                return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model configuration failed {suffix}"
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: model configuration failed"
        if message in SAFE_MODEL_GATEWAY_FAILURES:
            if suffix:
                return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {message} {suffix}"
            return f"{GENERIC_MODEL_GATEWAY_FAILURE}: {message}"
    return None


def safe_runtime_failure_reason(error: Exception, *, fallback: str = "runtime_failed") -> str:
    gateway_reason = safe_model_gateway_failure_reason(error)
    if gateway_reason is not None:
        return gateway_reason
    reason = normalize_failure_reason(str(error))
    if not is_safe_failure_reason(reason):
        return fallback
    return reason[:MAX_FAILURE_REASON_LENGTH]


def safe_runtime_failure_diagnostic(
    error: Exception,
    *,
    fallback: str = "runtime_failed",
) -> RuntimeFailureDiagnostic:
    """Return a redacted, structured diagnostic payload for failed runs."""

    reason = safe_runtime_failure_reason(error, fallback=fallback)
    status_code = error.status_code if isinstance(error, ModelTransportError) else None
    return runtime_failure_diagnostic_from_reason(reason, status_code=status_code)


def runtime_failure_diagnostic_from_reason(
    reason: str | None,
    *,
    status_code: int | None = None,
) -> RuntimeFailureDiagnostic:
    """Classify an already-redacted failure reason into UI/operator hints."""

    normalized = normalize_failure_reason(reason or "") or "runtime_failed"
    if not is_safe_failure_reason(normalized):
        normalized = "runtime_failed"
    if status_code is None:
        status_code = _status_code_from_reason(normalized)
    lowered = normalized.lower()

    diagnostic = _base_diagnostic(
        normalized,
        error_stage="runtime",
        error_category="internal",
        error_code="runtime.failed",
        retryable=False,
        suggested_action="查看运行详情中的上一条失败事件；如果失败重复出现，检查对应模式、工具或模型配置。",
    )

    crew_timeout = _CREWAI_STEP_TIMEOUT.fullmatch(normalized)
    if crew_timeout is not None:
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="crew_step",
            error_category="step_timeout",
            error_code="crew.step_timeout",
            retryable=True,
            suggested_action="CrewAI 子 Agent 步骤执行超时；可压缩该步骤输入、切换更快模型、提高审查/汇总步骤超时或降低并发等待。",
        )
        diagnostic["step_id"] = crew_timeout.group("step")
        diagnostic["actor"] = crew_timeout.group("actor")
    elif (checkpoint_mismatch := _MODEL_REQUEST_CHECKPOINT_MISMATCH.fullmatch(normalized)) is not None:
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime_checkpoint",
            error_category="model_request_changed",
            error_code="runtime.model_request_changed_after_checkpoint",
            retryable=False,
            suggested_action=(
                "运行恢复时同一步骤的模型请求发生变化；检查运行期间是否修改了模式、"
                "Agent、工具或模型配置，必要时重新提交任务。"
            ),
        )
        diagnostic["step_id"] = checkpoint_mismatch.group("step")
        diagnostic["actor"] = checkpoint_mismatch.group("actor")
        diagnostic["purpose"] = checkpoint_mismatch.group("purpose")
        diagnostic["call_index"] = int(checkpoint_mismatch.group("call_index"))
        diagnostic["expected_request_sha256"] = checkpoint_mismatch.group("expected")
        diagnostic["actual_request_sha256"] = checkpoint_mismatch.group("actual")
    elif (
        "model gateway failed" in lowered
        or "model response text is empty" in lowered
        or "model response is empty" in lowered
    ):
        diagnostic = _model_gateway_diagnostic(normalized, status_code=status_code)
    elif "model response budget is unverifiable" in lowered:
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime_accounting",
            error_category="model_usage_unverifiable",
            error_code="runtime.model_usage_unverifiable",
            retryable=True,
            suggested_action="模型返回内容可用但 usage 账本不可验证；系统会优先使用保守估算，若仍失败请检查模型适配器 usage 字段、输出长度和上下文预算。",
        )
    elif lowered.startswith("capability failed:"):
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="capability",
            error_category="execution_failed",
            error_code="capability.execution_failed",
            retryable=False,
            suggested_action="工具执行被拒绝或失败；查看工具失败事件中的字段校验摘要，修正参数后重试。",
        )
    elif normalized == "runtime_not_configured":
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime_configuration",
            error_category="missing_runtime",
            error_code="runtime.not_configured",
            retryable=False,
            suggested_action="当前运行模式没有可用运行时；请检查服务启动参数和模式注册配置后重试。",
        )
    elif normalized == "dispatch accounting exhausted":
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime_accounting",
            error_category="accounting_guardrail",
            error_code="runtime.dispatch_accounting_exhausted",
            retryable=False,
            suggested_action="派单运行的账本或预算记录已耗尽，系统已停止交付；请检查任务预算、模型用量记录和产物提交记录。",
        )
    elif normalized == "artifact rollback failed":
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="artifact_storage",
            error_category="rollback_failed",
            error_code="artifact.rollback_failed",
            retryable=False,
            suggested_action="产物写入回滚失败；请检查文件存储/对象存储状态，确认残留产物后再重试。",
        )
    elif lowered in {"unaccounted_usage", "model_outcome_uncertain", "tool_outcome_uncertain"}:
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime_accounting",
            error_category="accounting_guardrail",
            error_code=f"runtime.{lowered}",
            retryable=False,
            suggested_action="运行结果账本不完整，系统已阻止继续交付；请检查模型/工具结果记录链路后重试。",
        )
    elif "timeout" in lowered or "timed out" in lowered:
        diagnostic = _base_diagnostic(
            normalized,
            error_stage="runtime",
            error_category="timeout",
            error_code="runtime.timeout",
            retryable=True,
            suggested_action="任务运行超时；可缩小任务范围、降低并发或稍后重试。",
        )
    return diagnostic


def _model_gateway_diagnostic(
    reason: str,
    *,
    status_code: int | None,
) -> RuntimeFailureDiagnostic:
    lowered = reason.lower()
    if "model response text is empty" in lowered or "model response is empty" in lowered:
        return _with_model_context(
            _base_diagnostic(
            reason,
            error_stage="model_gateway",
            error_category="empty_response",
            error_code="model.empty_response",
            retryable=True,
            suggested_action="模型返回了空内容；系统可重试、切换备用模型，或压缩上下文后重新执行该步骤。",
            status_code=status_code,
            ),
            reason,
        )
    if "no capable deployment" in lowered:
        return _base_diagnostic(
            reason,
            error_stage="model_routing",
            error_category="no_capable_model",
            error_code="model.no_capable_deployment",
            retryable=False,
            suggested_action="在模型配置中为当前能力补齐可用部署，或调整 Agent/工作流使用的逻辑模型。",
        )
    if "capacity queue is full" in lowered:
        return _base_diagnostic(
            reason,
            error_stage="model_capacity",
            error_category="queue_full",
            error_code="model.capacity_queue_full",
            retryable=True,
            suggested_action="模型并发队列已满；稍后重试，或降低并发/切换备用模型。",
        )
    if "capacity queue timeout" in lowered:
        return _base_diagnostic(
            reason,
            error_stage="model_capacity",
            error_category="queue_timeout",
            error_code="model.capacity_timeout",
            retryable=True,
            suggested_action="模型容量等待超时；稍后重试，或增加可用席位/切换备用模型。",
        )
    if "capacity configuration failed" in lowered:
        return _base_diagnostic(
            reason,
            error_stage="model_capacity",
            error_category="configuration",
            error_code="model.capacity_configuration_failed",
            retryable=False,
            suggested_action="检查模型容量配置、配额作用域和并发参数，修复后重新发布配置。",
        )
    if "capacity backend failed" in lowered:
        return _base_diagnostic(
            reason,
            error_stage="model_capacity",
            error_category="backend",
            error_code="model.capacity_backend_failed",
            retryable=True,
            suggested_action="检查容量后端服务状态；恢复后可重试本次任务。",
        )
    if "capacity unavailable" in lowered:
        diagnostic = _base_diagnostic(
            reason,
            error_stage="model_capacity",
            error_category="unavailable",
            error_code="model.capacity_unavailable",
            retryable=True,
            suggested_action="当前模型容量不可用；稍后重试，或切换到可用模型。",
        )
        context = _CAPACITY_CONTEXT.search(reason)
        if context is not None:
            logical_models = context.group("logical_models")
            deployments = context.group("deployments")
            diagnostic["logical_models"] = context.group("logical_models")
            diagnostic["deployments"] = context.group("deployments")
            diagnostic["suggested_action"] = (
                f"当前模型容量不可用：{logical_models}；候选部署：{deployments}。"
                "可稍后重试、降低并发，或切换到可用模型。"
            )
        return diagnostic
    if "configuration failed" in lowered:
        return _with_model_context(
            _base_diagnostic(
            reason,
            error_stage="model_configuration",
            error_category="configuration",
            error_code="model.configuration_failed",
            retryable=False,
            suggested_action="检查模型 API Key、API Base、模型名和供应商权限后重试。",
            status_code=status_code,
            ),
            reason,
        )
    if "response failed" in lowered:
        return _with_model_context(
            _provider_diagnostic(reason, status_code=status_code, response=True),
            reason,
        )
    if "transport failed" in lowered:
        return _with_model_context(
            _provider_diagnostic(reason, status_code=status_code, response=False),
            reason,
        )
    return _with_model_context(
        _base_diagnostic(
        reason,
        error_stage="model_gateway",
        error_category="gateway",
        error_code="model.gateway_failed",
        retryable=True,
        suggested_action="模型网关调用失败；查看模型配置与调用错误日志，必要时切换备用模型后重试。",
        status_code=status_code,
        ),
        reason,
    )


def _provider_diagnostic(
    reason: str,
    *,
    status_code: int | None,
    response: bool,
) -> RuntimeFailureDiagnostic:
    if status_code == 400:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="bad_request",
            error_code="model.provider_bad_request",
            retryable=False,
            suggested_action="模型请求参数被供应商拒绝；检查模型名、上下文长度、工具参数和供应商兼容性。",
            status_code=status_code,
        )
    if status_code in {401, 403}:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="authentication",
            error_code="model.provider_auth_failed",
            retryable=False,
            suggested_action="模型供应商拒绝请求；检查 API Key、Base URL、模型权限和账号额度后重试。",
            status_code=status_code,
        )
    if status_code == 402:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="quota_or_billing",
            error_code="model.provider_quota_or_billing_failed",
            retryable=False,
            suggested_action="模型账号额度或计费状态不可用；检查供应商余额、额度、账单或组织权限后重试。",
            status_code=status_code,
        )
    if status_code == 404:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="model_not_found",
            error_code="model.provider_model_not_found",
            retryable=False,
            suggested_action="供应商找不到该模型或端点；检查模型名、API Base 路径和部署配置。",
            status_code=status_code,
        )
    if status_code == 413:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="payload_too_large",
            error_code="model.provider_payload_too_large",
            retryable=False,
            suggested_action="模型请求体或上下文过大；压缩上下文、减少附件/历史消息或降低输出长度后重试。",
            status_code=status_code,
        )
    if status_code == 429:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="rate_limited",
            error_code="model.provider_rate_limited",
            retryable=True,
            suggested_action="模型供应商限流；稍后重试，或降低并发/切换备用模型。",
            status_code=status_code,
        )
    if status_code is not None and status_code >= 500:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="upstream_unavailable",
            error_code="model.provider_unavailable",
            retryable=True,
            suggested_action="上游模型服务异常；稍后重试，或临时切换备用模型。",
            status_code=status_code,
        )
    if status_code in {408, 409, 425}:
        return _base_diagnostic(
            reason,
            error_stage="model_provider",
            error_category="transient",
            error_code="model.provider_transient_failed",
            retryable=True,
            suggested_action="上游模型请求临时失败；可直接重试或切换备用模型。",
            status_code=status_code,
        )
    category = "invalid_response" if response else "transport"
    code = "model.provider_response_failed" if response else "model.provider_transport_failed"
    action = (
        "模型返回结构不符合系统契约；检查模型适配器、返回格式和供应商兼容性。"
        if response
        else "模型网络或上游连接失败；检查 API Base、网络连通性和供应商状态。"
    )
    return _base_diagnostic(
        reason,
        error_stage="model_provider",
        error_category=category,
        error_code=code,
        retryable=response is False,
        suggested_action=action,
        status_code=status_code,
    )


def _base_diagnostic(
    reason: str,
    *,
    error_stage: str,
    error_category: str,
    error_code: str,
    retryable: bool,
    suggested_action: str,
    status_code: int | None = None,
) -> RuntimeFailureDiagnostic:
    diagnostic: RuntimeFailureDiagnostic = {
        "error_summary": reason[:MAX_FAILURE_REASON_LENGTH],
        "error_stage": error_stage,
        "error_category": error_category,
        "error_code": error_code,
        "retryable": retryable,
        "suggested_action": suggested_action,
        "possible_cause": _possible_cause(
            error_stage=error_stage,
            error_category=error_category,
            status_code=status_code,
        ),
    }
    if status_code is not None:
        diagnostic["status_code"] = status_code
    return diagnostic


def _model_context_suffix(error: Exception, *, status_code: int | None = None) -> str:
    raw_logical_models = getattr(error, "logical_models", ())
    raw_deployments = getattr(error, "deployments", ())
    if not isinstance(raw_logical_models, tuple) or not isinstance(raw_deployments, tuple):
        return ""
    logical_models = ",".join(item for item in raw_logical_models if isinstance(item, str))
    deployments = ",".join(item for item in raw_deployments if isinstance(item, str))
    if (
        not logical_models
        or not deployments
        or _SAFE_MODEL_CONTEXT_VALUE.fullmatch(logical_models) is None
        or _SAFE_MODEL_CONTEXT_VALUE.fullmatch(deployments) is None
    ):
        return ""
    if status_code is not None:
        return f"(status={status_code}; logical_models={logical_models}; deployments={deployments})"
    return f"(logical_models={logical_models}; deployments={deployments})"


def _with_model_context(
    diagnostic: RuntimeFailureDiagnostic, reason: str
) -> RuntimeFailureDiagnostic:
    context = _MODEL_CONTEXT.search(reason)
    if context is None:
        return diagnostic
    logical_models = context.group("logical_models")
    deployments = context.group("deployments")
    diagnostic["logical_models"] = logical_models
    diagnostic["deployments"] = deployments
    diagnostic["suggested_action"] = (
        f"{diagnostic['suggested_action']} 相关模型：{logical_models}；相关部署：{deployments}。"
    )
    return diagnostic


def _possible_cause(
    *,
    error_stage: str,
    error_category: str,
    status_code: int | None,
) -> str:
    if error_stage == "model_provider":
        if error_category == "authentication":
            return "API Key 失效、模型权限不足、供应商账号或 Base URL 配置不匹配。"
        if error_category == "payload_too_large":
            return "多轮历史、附件或中间产物进入模型请求后超过供应商请求体或上下文限制。"
        if error_category == "rate_limited":
            return "该模型部署被供应商限流，或同一账号/部署并发与 TPM/RPM 达到上限。"
        if error_category == "upstream_unavailable":
            return "供应商服务异常、部署临时不可用、网络链路中断或网关到上游超时。"
        if error_category == "model_not_found":
            return "模型名、部署名或 API Base 路径与供应商实际端点不一致。"
        if error_category == "bad_request":
            return "请求参数、工具格式、上下文内容或该供应商兼容层不接受当前 payload。"
        if status_code is None:
            return "网络连接失败、供应商连接被重置、DNS/TLS/代理异常，或上游未返回可解析状态码。"
    if error_stage == "model_gateway":
        if error_category == "empty_response":
            return "模型返回空文本、被上游截断、上下文过大导致响应异常，或该部署与当前模式不兼容。"
        return "模型网关调度、fallback、结果记录或模型返回契约出现异常。"
    if error_stage == "model_capacity":
        return "目标模型部署并发已满、容量租约后端不可用、容量配置错误或健康状态被标记不可用。"
    if error_stage == "model_routing":
        return "当前 Agent/工作流要求的能力没有匹配到已启用模型部署。"
    if error_stage == "model_configuration":
        return "模型 API Key、模型名、Base URL、供应商权限或容量配置不完整。"
    if error_stage == "runtime_checkpoint":
        return "任务恢复或重试期间，步骤输入、工具列表、模型选择或配置版本发生变化，导致账本无法安全复用。"
    return "查看同一运行中上一条失败事件、模型事件、工具事件和配置变更记录来定位具体层级。"


def _status_code_from_reason(reason: str) -> int | None:
    match = _STATUS_CODE.search(reason)
    if match is None:
        return None
    return int(match.group("status"))


def normalize_failure_reason(reason: str) -> str:
    return " ".join(reason.strip().split())


def is_safe_failure_reason(reason: str) -> bool:
    return bool(reason) and SENSITIVE_FAILURE_REASON.search(reason) is None


def is_legacy_generic_failure_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return normalize_failure_reason(reason) in LEGACY_GENERIC_FAILURES


__all__ = [
    "GENERIC_MODEL_GATEWAY_FAILURE",
    "MAX_FAILURE_REASON_LENGTH",
    "SENSITIVE_FAILURE_REASON",
    "is_legacy_generic_failure_reason",
    "is_safe_failure_reason",
    "normalize_failure_reason",
    "runtime_failure_diagnostic_from_reason",
    "safe_model_gateway_failure_reason",
    "safe_runtime_failure_diagnostic",
    "safe_runtime_failure_reason",
]
