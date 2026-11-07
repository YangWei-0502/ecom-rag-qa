"""企业级 RAG HTTP 服务（FastAPI）：知识管理 + 检索 + 问答。

核心业务逻辑全部复用 app/ 服务层（knowledge / chat），本包只做
HTTP 协议适配、校验、中间件与错误信封。
"""
