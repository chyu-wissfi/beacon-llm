# CONTEXT.md - LLM Gateway 术语表

本文件是项目统一语言的唯一权威。代码、文档、spec、提交信息中的术语与本表冲突时，以本表为准并当场修正。本表只收术语，不收实现细节。

## 领域一句话

本项目是一个 LLM Gateway：业务 Agent 的唯一模型出口，集中管理契约、路由、校验、预算与审计。

## 术语

### 调用方（Caller）
通过 API Key 标识的业务 Agent 或个人助手，是配额、审计与一切聚合口径的归属主体。本项目**无多租户概念**，"租户"一词禁用，凡聚合维度一律指调用方。

### 平台模型（Platform Model）
调用方请求中使用的逻辑模型名（如 `general-primary`），由网关映射到供应商模型。平台模型表即白名单，未知名称直接拒绝，不存在隐式降级。

### 供应商模型（Provider Model）
上游供应商处的真实模型标识（如 `deepseek-chat`），只存在于网关内部，调用方不可见。

### Provider（Adapter）
供应商适配器，实现 Provider Protocol，把内部 ModelRequest 翻译为特定供应商的协议调用。网关内所有上游协议差异终止于此层。

### ModelRequest
外部请求经 API 层规范化后的内部统一契约。进入编排层之后，全链路只出现这一种请求类型。

### RunContext（运行上下文）
准入时一次性构建的不可变对象：渲染完成的 Prompt、选定的 Schema 与 Validation Profile、预算实例、request_id、Trace 骨架。在途回合不受模板热加载或配置变更影响。

### Run 预算（Run Budget）
单次调用中一切再尝试形式（重试、Fallback、修复调用）共享的消耗上限：总尝试次数 + 墙钟超时。网关是重试的唯一权威，供应商 SDK 内置重试一律关闭。

### 尝试（Attempt）
对某个候选端点的一次真实上游调用，预算按尝试计数。

### Fallback 链（Fallback Chain）
每个平台模型声明式配置的候选端点序列，按序尝试。

### 路由理由（Route Reason）
Router 求值 Fallback 链时为每个决定留下的解释（如"主模型连续失败触发降级"），随 Trace 落库。

### 修复调用（Repair Call）
校验失败后携带错误反馈对同一端点的重调；截断场景改为提高 max_tokens 重调。消耗 Run 预算。

### 终态（Terminal State）
一次调用恰好迁移一次的最终状态：`success` / `failed` / `cancelled` 三选一。

### Trace（调用追踪）
一次调用的审计记录，含 usage、成本、延迟、尝试数、路由理由、TTFT、价格版本、Prompt 与校验档案版本，**永不记录消息内容**。

### Prompt 模板（Prompt Template）
由 name + version 定位的系统提示词资产，版本即文件。调用方只能选择模板与变量，不能提交模板正文。

### Validation Profile（校验档案）
由 name + version 定位的业务规则模型，在结构校验通过之后执行，拦截"结构合法但业务非法"的输出。

### 准入控制（Admission Control）
请求进入编排前的多层拦截：鉴权、全局并发、每供应商并发、每模型 RPM；TPM 采用调用后按实际用量记账。

### 熔断器（Circuit Breaker）
按模型的失败保护：连续失败达到阈值即熔断一段时间，半开状态放行单个探测请求验证恢复。

### Fake Adapter
剧本化的供应商实现，可稳定复现成功、限流、超时、流中断、非法输出，是行为测试的地基。

### ModelPort
调用方（Agent 侧）依赖的内部客户端包，封装网关接入细节并映射错误码。Agent 不导入任何供应商 SDK。

### TTFT（首 Token 延迟）
流式调用从准入到第一个内容块之间的时长，流式质量的核心指标。
