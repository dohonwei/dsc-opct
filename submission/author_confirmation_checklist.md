# BSPC 投稿前作者确认清单

版本日期：2026-09-16

以下项目不能通过代码、统计分析或旧论文推断，必须由作者确认。除这些项目和尚未取得的预留外部数据外，当前审稿闭环中的可执行分析与文档检查均已完成。

## 必须在投稿前完成

- [x] **公开代码入口**：已发布 v22 校验归档、源代码、聚合结果和验证报告，并核对远端 main 提交、SHA-256 旁车文件与本地上传归档。入口：https://github.com/dohonwei/dsc-opct。
## 已由原始记录核验

- [x] **Funding**：正式国自然计划书与批准通知确认项目号 `62172081`，同一 EPPVR 论文也使用该项目号。当前稿件已加入 `National Natural Science Foundation of China (No. 62172081)`；未带入缺少当前研究关联证据的广西项目。
- [x] **伦理单位英文名**：同一 EPPVR 论文针对审批号 `106142023122227999` 使用 `Ethical Committee of the University of Electronic Science and Technology of China`，主文与补充材料已统一。
- [x] **Channels 5--9 顺序**：已检查原始数据目录、交接代码和既有采集处理链。记录只能确认其集合组成（2 EOG、1 PPG、1 GSR、1 temperature），不能恢复逐索引顺序；正文与补充材料已明确说明 unavailable，且这些通道不进入本研究分析。

## 外部科学边界

- [x] **EKM-ED 前瞻性 one-shot**：已在参与者数值访问前完成数据预留、校验和核验、header-only schema 绑定和实现锁定；冻结端点产生 0 名合格参与者，流程在模型拟合前停止。该结果是前瞻性 endpoint-transport failure，不是有效性、无害性或安全性证据。
- [ ] **更强声明所需的后续外部试验**：AMIGOS 或 Emognition 获得正式授权后，可继续执行已锁定的 one-shot 协议。此项不是当前有界 BSPC 稿件的投稿阻断项，但在外部数据通过全部冻结门控并产生非 identity action 前，不得升级为 `prospective external effectiveness`、`universally safe transfer`、`externally validated safety` 或等价措辞。

## 投稿系统核对

- [x] 稿件中的作者顺序、单位、邮箱及通讯作者已与作者提供的正式信息逐项核对；投稿系统录入时仍需按同一顺序填写。
- [x] Cover letter、Highlights、主文和补充材料中的 claim boundary 已通过打包校验保持一致。
- [x] 主文 PDF 与补充 PDF 均由当前源文件重新编译，并将收入 v22 包，而不是旧快照。
- [x] 打包器排除了受限原始信号文件和指定的参与者级治理敏感输出；公开范围仅包括代码、聚合结果和许可允许的派生物。
