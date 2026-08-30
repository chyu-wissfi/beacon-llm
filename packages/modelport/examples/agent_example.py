"""示例 Agent：只依赖 modelport 的最小调用方（M12 任务 3 的证明载体）。

隔离测试（tests/contract/test_modelport.py）解析本文件的 import 面，断言
Agent 代码不出现任何供应商 SDK（openai / anthropic）导入——"Agent 不依赖
供应商 SDK"是可验收行为，不是文档口号（design.md §4.7）。

运行：`DEEPSEEK_API_KEY=... python examples/agent_example.py`
（需要先起网关：make run；api_key 取 config/callers.yaml 的签发条目。）
"""

import asyncio
import os

import modelport


async def main() -> None:
    port = modelport.ModelPort(
        base_url=os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000"),
        api_key=os.environ["CALLER_API_KEY"],
    )
    try:
        # 非流式：result.request_id 可与 /v1/traces 对账。
        result = await port.complete(
            model="general-primary",
            messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        )
        print(f"[{result.request_id}] {result.content}")

        # 流式：逐块消费增量。
        async for chunk in port.stream(
            model="general-primary",
            messages=[{"role": "user", "content": "数到五"}],
        ):
            print(chunk.delta, end="", flush=True)
        print()
    except modelport.RateLimitedError as exc:
        print(f"被限流（{exc.code}），建议 {exc.retry_after} 秒后重试")
    except modelport.ModelPortError as exc:
        print(f"调用失败（{exc.code}）：{exc.message}")
    finally:
        await port.aclose()


if __name__ == "__main__":
    asyncio.run(main())
