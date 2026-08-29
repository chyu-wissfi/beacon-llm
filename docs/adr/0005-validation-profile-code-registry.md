# ADR-0005: Validation Profile 采用 Pydantic 代码注册表，而非声明式规则资产

日期：2026-08-30 · 状态：已接受

## 背景

结构校验（JSON Schema）表达不了业务规则：输出可能"每个字段都类型正确，但两个互斥字段同时出现"。需要在结构校验之后执行业务校验。业务规则的供给方式有三种候选。

## 决策

Validation Profile = `src/llm_gateway/validation/` 下的 Pydantic 模型，带 name + version，业务规则写在 `model_validator` 里。调用方请求扩展字段 `validation: {name, version}` 选择；未注册 -> 400 `unknown_validation_profile`（调用模型前失败）。新增规则 = 改代码、走测试、发版。

## 备选方案

- **声明式 YAML 规则资产**（像 Prompt 模板一样热加载）：规则词汇表必然受限（mutually_exclusive / requires / at_least_one_of...），真实业务规则形态开放（条件依赖、跨字段计算、引用外部约束），迟早不够用，然后被迫在 YAML 里发明一门小语言——那是真正的深坑。否决。
- **扩展 JSON Schema 关键字**（`x-mutually-exclusive`）：把语义塞进结构描述工具，供应商侧会忽略这些扩展关键字，双重校验名存实亡。否决。

## 理由

业务规则本身就该有类型、有单测、走 CI——代码注册表天然获得这一切。代价是"新增规则要发版"，但与调用方管理（改 callers.yaml 重启生效）处于同一运维粒度，1~5 个内部调用方场景下不构成瓶颈。

## 后果

正面：任意复杂度规则；规则即代码可 pytest；与 Prompt 模板形成清晰分工——模板是数据（正文易变、热加载），校验是逻辑（规则需要测试、随版本发布）。
负面：不能热加载新规则；调用方引入新 Profile 依赖网关发版节奏。
