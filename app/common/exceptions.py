class AppException(Exception):
    code: int = 500
    msg: str = "服务内部错误"

    def __init__(self, msg: str | None = None):
        super().__init__(msg or self.msg)
        if msg:
            self.msg = msg


class ParamException(AppException):
    code = 400
    msg = "参数错误"


class UserAuthException(AppException):
    code = 401
    msg = "用户身份非法，请重新获取用户标识"


class PermissionDeniedException(AppException):
    code = 403
    msg = "无权限操作该数据"


class NotFoundException(AppException):
    code = 404
    msg = "资源不存在"


class ServiceUnavailableException(AppException):
    code = 503
    msg = "当前服务暂不可用，请稍后重试"


class InputComplianceException(AppException):
    code = 601
    msg = "输入内容违规"


class OutputComplianceException(AppException):
    code = 602
    msg = "输出内容违规"


class OutOfScopeException(AppException):
    code = 603
    msg = "功能越界请求"


class LLMCallException(AppException):
    code = 503
    msg = "大模型调用失败"
