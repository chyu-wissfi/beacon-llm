"""openai 3.6.0 兼容 shim：让 respx 能拦截上游流量（M04 从 contract conftest 提取）。

openai 3.x 默认用 httpx2.AsyncClient 发请求，而 respx 0.23.1 只 patch
httpx/httpcore 层，对 httpx2 流量完全不可见——mock 会静默漏到真实网络（已实测
DeepSeek 返回 401）。openai 3.x 官方支持运行 legacy httpx client（http_client
参数的 is_legacy_* 分支），因此这里把 openai 的默认 client 工厂替换为 legacy
httpx1 版本，使 respx 恢复拦截。该 shim 只动 openai 库自己的命名空间，不碰
网关代码；升级 openai 或 respx 时需复核此 shim 是否仍必要/仍有效。

anthropic SDK 用的是 httpx1，无需本 shim（respx 原生可拦）。
"""

from typing import Any, cast

import httpx
import pytest


@pytest.fixture
def openai_legacy_httpx(monkeypatch):
    # 只透传 base_url：客户端级 timeout 对离线 mock 无意义，且 openai 传入的
    # httpx2.Timeout 对象与 httpx1 不兼容，直接忽略（网关的每请求超时由
    # openai 的 legacy 归一化逻辑另行处理，与本工厂无关）。
    def _legacy_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
        # cast 是运行期恒等：仅满足 httpx base_url: URLTypes 的标注。openai 构造
        # AsyncHttpxClientWrapper 时恒传 base_url（openai/_base_client.py），不会缺键。
        return httpx.AsyncClient(
            base_url=cast("httpx.URL | str", kwargs.get("base_url")),
        )

    monkeypatch.setattr("openai._base_client.AsyncHttpxClientWrapper", _legacy_httpx_client)
