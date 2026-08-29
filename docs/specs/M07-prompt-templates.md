# M07 - Prompt 模板资产化与热加载

## 目标

模板从代码内 dict 迁移到 `templates/<name>/<version>.yaml`（版本即文件）；mtime 惰性热加载；缺变量在调用模型前失败；加载失败保留旧版。

## 前置依赖

M06。

## 任务

1. `templates/knowledge_decision/v1.yaml`（迁移 demo 模板，含 name/version/system_template 元数据）。
2. `prompt/loader.py`：启动全量加载；每次请求检查目录 mtime，变更才重载（惰性）。单文件解析失败：**保留旧版本继续服务** + error 日志（错误含文件路径与原因）。
3. `services/prompt_service.py`：`render_prompt(selection) -> Message`；模板不存在 400 `unknown_prompt_template`；缺变量 400 `missing_prompt_variable`；渲染结果进 RunContext（M06 已留位）。
4. 测试：
   - 请求携带 `prompt: {name, version, variables}` 经 extra_body 到达，渲染正确、系统消息注入在首位。
   - 缺变量：响应 400 且 **上游请求数 == 0**（调用模型前失败）。
   - 热加载：启动后磁盘上新增 v2 模板文件，下一次请求立即可用（无需重启）。
   - 坏文件：写入非法 yaml，下一次请求仍用旧版成功 + 日志含 error。
   - 调用方不能提交模板正文（extra=forbid 已保证，加断言测试）。

## 验收

```bash
make check
uv run pytest tests/unit/prompt/ -q
uv run pytest tests/contract/test_prompt.py -q -k "missing_variable_or_hot_reload"
# 调用模型前失败证明：缺变量场景上游请求数 == 0
uv run pytest tests/contract/test_prompt.py -q -k fail_before_upstream
```

## 覆盖的不变量

- #11（Prompt 缺变量在调用模型前失败）
- #1（模板资产治理）

## 边界

- 模板只有 system 角色正文；不做模板变量类型校验（全部按字符串替换，与 demo 一致）。
