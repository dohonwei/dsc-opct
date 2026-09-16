from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_nature_review_20260910"
DOCX = OUT / "review_dcs_opct_v11_20260910.docx"
MARKDOWN = OUT / "review_dcs_opct_v11_20260910.md"


EDITOR_PARAGRAPHS = [
    (
        "Major revision。稿件围绕重复测量 EEG 情绪识别中的身份暴露风险，建立了从反事实剂量、风险预测到选择性概率校准的完整工程链条。"
        "冻结规则、保留失败记录、GPU 复现与非干预边界处理均明显强于常见算法对比稿。当前主要障碍不是实现完整性，而是外部有效性和推断层级仍不足以支撑“有效的安全跨域迁移框架”这一强结论。"
    ),
    (
        "现有结果可以支撑一个有价值但更窄的结论：在已检查的开发域内，配置级身份乐观风险能够被选择性校准，并可在不满足门控条件时保持原概率。"
        "要达到一区论文的说服力，必须完成至少一个预先保留、规则冻结且产生非 identity action 的外部有效释放；同时需要收紧统计推断、补全 OPCT 优化目标，并重构主文以突出唯一科学问题。"
    ),
]


REVIEWERS = [
    {
        "role": "审稿人 1 机器学习方法与创新性",
        "overall": (
            "稿件不再只是信号处理组合比较，而是提出了面向评估风险的三阶段决策框架。其价值来自问题分解和受限释放机制，而不是 CORAL、quantile mapping 或 Platt 型映射本身。"
            "然而，当前创新性论证仍主要依靠组件串联，尚未清楚证明完整框架相对于最接近的 selective calibration、risk-controlled adaptation 和 abstention 方法新增了什么可验证能力。"
        ),
        "major": [
            (
                "最接近方法边界尚未建立",
                "Introduction 仅分别引用 calibration、CORAL 和 covariate-shift 文献，没有以统一问题表比较现有方法是否同时具备配置级风险端点、无标签适用性门控、有限标签证书、顺序保持和 identity fallback。审稿人仍可能把 DCS-OPCT 视为已知组件的工程组合。需要增加 closest-method taxonomy，并将创新表述改为可检验的能力差异，而非组件数量。",
                "Introduction 第 4 至第 6 段及 Contributions 段",
            ),
            (
                "OPCT 优化目标描述不完整",
                "Methods 给出了正斜率映射和训练轮数，却没有给出实际优化目标。代码显示映射以 identity probability 的 logit 为输入，以经 CORAL 或 quantile transform 得到的 base probability 为软目标，最小化均方差并对 scale 偏离 1 和 shift 加正则。缺少该方程会使方法无法从论文独立复现，也容易被误解为使用了目标标签。",
                "Methods Stage III 段落及 develop_order_preserving_transport_v6.py 的 fit_opct",
            ),
            (
                "witness 的独立性措辞过强",
                "CORAL 分支和 quantile 分支使用同一目标配置表、同一冻结风险模型并共享下游规则，因此不是统计独立证据。应统一改为 separately constructed 或 complementary unlabeled witness，并明确 agreement gate 只检验方向一致性，不能证明独立验证或因果不变性。",
                "Abstract、Introduction、Methods Stage III、Discussion",
            ),
        ],
        "minor": [
            "说明 CUDA 优化的必要性是计算实现选择，而不是方法成立条件；同时报告 CPU 等价性边界。",
            "将 identity probability 与 identity fallback 的术语区分为 original frozen probability 和 non-intervention action，避免读者混淆。",
        ],
        "confidence": "高",
    },
    {
        "role": "审稿人 2 统计与推断",
        "overall": (
            "作者对 dose cluster 依赖、bootstrap 单位和安全上界进行了大量补强，且明确保留了七域鲁棒性门控失败，这是重要优点。"
            "但训练域数量、重复 audit 的解释以及跨配置相关性仍限制统计结论。当前 p-value 和置信区间应被限定为对固定开发数据和固定 held-out halves 的算法敏感性推断。"
        ),
        "major": [
            (
                "风险模型仅有两个独立训练域",
                "Stage II 的 480 行并不等价于 480 个独立跨域样本；真正独立的数据集层级只有 DEAP 和 MAHNOB-HCI 两个。两次 leave-dataset-out 结果无法稳定估计域间异质性，也不足以证明 predictor direction 可泛化。应把数据集层级样本量显式写入 Abstract、Methods 和 Limitations，并避免用 configuration n 掩盖 domain n。",
                "Abstract Stage II 结果、Methods 风险模型、Results 风险迁移",
            ),
            (
                "重复 audit 不是独立实验重复",
                "100 个 seeded audit orderings 共用相同目标数据和固定 held-out half。paired bootstrap、sign randomization 和 McNemar 结果只能描述固定数据条件下的分配敏感性，不能产生未来域层级的频率学证据。建议将主文显著性叙事降为 conditional design comparison，并把完整多重检验表放入补充材料。",
                "Methods Matched audit-budget decision analysis 与 Results budget curves",
            ),
            (
                "剩余多层相关性未充分建模",
                "完整四剂量 cluster 修正了最明显的伪重复，但同一 dataset-task-seed 下的 representations 和 models 仍共享原始试次、划分和标签。ICC(1,1) 只覆盖 cluster 内四个剂量。建议增加按 dataset-task-seed 或原始 split family 聚合的保守敏感性分析，或明确说明当前区间未覆盖这些更高层相关性。",
                "Methods audit-yield analysis、Supplementary complete-cluster section",
            ),
            (
                "post-hoc reporting cell 容易被理解为结果驱动选择",
                "0.02 阈值虽已冻结，但 20% audit budget 被称为 designated post hoc cell。精确 p=0.000345 不应被呈现为预设验证结果。建议把它放在 Supplementary，并在主文仅报告效应量、随机基线和敏感性范围。",
                "Results engineering-anchor paragraph 与 Supplementary audit-yield table",
            ),
        ],
        "minor": [
            "统一 one-sided familywise lower bound、Clopper-Pearson upper bound 和 bootstrap interval 的置信水平与估计单位。",
            "解释 balanced accuracy 阈值 0.60 和 Brier gain 0.001 的工程来源，并将经验冻结与外部有效性区分。",
        ],
        "confidence": "高",
    },
    {
        "role": "审稿人 3 外部有效性与生物医学信号",
        "overall": (
            "稿件对不同数据集的证据角色划分非常谨慎，AVDOS-VR、FACED 和 EEGEmotions-27 均未被错误包装为成功确认。"
            "但这也意味着当前没有一次预先保留、非 identity、通过全部冻结标准的外部有效释放。对于强调 cross-domain 和 safe transfer 的一区投稿，这仍是决定性缺口。"
        ),
        "major": [
            (
                "缺少成功的前瞻外部有效释放",
                "EPPVR、CASE 和 CEAP 已参与 DCS-OPCT 开发；AVDOS-VR 和 FACED 为 post-access exploratory；EEGEmotions-27 仅支持 pre-signal abstention boundary。当前证据不能证明框架在未见兼容域上既能识别可干预情形又能产生有效校准。AMIGOS 或 Emognition 的冻结 one-shot 结果必须完整保留，无论成功、identity 或不可估计。",
                "Abstract、Study endpoint and evidence roles、Limitations",
            ),
            (
                "跨数据集 endpoint 语义并不完全一致",
                "部分数据集使用 experienced valence/arousal，EEGEmotions-27 使用 stimulus-category polarity，AVDOS-VR 使用重新组织的 arousal 类别。虽然论文已声明证据边界，但仍需用一张 endpoint harmonization 表说明每个域的标签来源、阈值、排除规则和可比较范围。否则 domain shift 可能混合生理差异与标签构念差异。",
                "Methods evidence roles 与 Supplementary evidence-role matrix",
            ),
            (
                "生物医学信号处理信息被框架叙事压缩",
                "主文没有集中列出各数据集参与人数、通道、采样率、窗口、标签和试次数，也缺少 EPPVR 自采数据的伦理审批编号、知情同意及基线处理说明。对 BSPC 或更高水平生物医学信号期刊，这些信息是可评估性要求，而不是附属细节。",
                "Materials and methods 与 Data and code availability",
            ),
        ],
        "minor": [
            "明确本框架的目标用户是 benchmark designer，而非穿戴式情绪识别终端用户。",
            "将 wearable 或 clinical implication 限制在审计流程，不要暗示直接改善个体情绪预测。",
        ],
        "confidence": "高",
    },
    {
        "role": "审稿人 4 可复现性与工程验证",
        "overall": (
            "冻结哈希、失败版本保留、GPU 测试、独立验证和到位门控构成了异常完整的工程证据。"
            "目前的主要问题是这些证据分散在多个版本和目录中，论文正文的计数已经落后于最新 177 文件、22 项 inventory validation 和 6 项 release readiness。投稿包需要单一入口和不可歧义的证据清单。"
        ),
        "major": [
            (
                "缺少面向审稿人的单一复现入口",
                "论文列举大量 validator 数量，但没有给出从环境检查、冻结验证、表图复建到最终 PDF 的一个命令或分阶段命令。建议提供 submission-level driver、固定环境文件、预计 GPU 时间和磁盘需求，并区分可公开数据与受限数据步骤。",
                "Methods Computational controls 与 Data and code availability",
            ),
            (
                "正文复现计数与当前权威状态不一致",
                "预览仍描述较早的 inventory 范围；当前权威清单已覆盖 177 个文件并通过 22/22，release audit 为 6/6，外部 arrival detector 为 7/7。需要由生成脚本自动读取报告，而不是手工维护数字，避免下一次迭代再次漂移。",
                "Methods Computational controls 与 Supplementary reproducibility section",
            ),
            (
                "受限数据的可审计复现边界需更具体",
                "AMIGOS 和 Emognition 的预访问软件测试很充分，但第三方审稿人无法验证真实数据 one-shot。应说明哪些 schema manifest、hash、aggregate outcomes 和 failure receipts 可公开，以及如何在不公开受试者值时验证 action 在标签访问前已锁定。",
                "Data availability、prospective external family protocol",
            ),
        ],
        "minor": [
            "报告关键软件版本、CUDA/cuDNN、deterministic flags 和硬件精度设置。",
            "将 v1 至 v11 的完整失败历史移至可检索补充表，主文仅保留与最终设计直接相关的三至四个转折。",
        ],
        "confidence": "高",
    },
    {
        "role": "审稿人 5 图表与写作清晰度",
        "overall": (
            "新版图形的可读性较早版本明显改善，且图注明确区分 non-intervention、exploratory 和 effectiveness。"
            "然而 36 页主文仍承载过多开发历史、诊断和外部失败分支，核心结果容易被淹没。若目标是一区，主文必须围绕一个问题、一个主要端点和一个外部判定流程组织。"
        ),
        "major": [
            (
                "主文信息负载过高",
                "Results 同时呈现 dose、ranking、risk transport、七域失败、六域 action、十类 calibrator、两个 post-access stress tests 和 EEGEmotions-27。建议主文保留三阶段主图、Stage I consequence、Stage II transfer、Stage III release/budget 和 external boundary 五个核心视觉单元，其余移入 Supplementary。",
                "Results 全节及 36 页 manuscript preview",
            ),
            (
                "安全措辞仍可能触发过度主张质疑",
                "小节标题 safety--efficacy tradeoff、framework 图中的 mechanism--risk--safety loop，以及若干 safe transfer 表述会先于限定语进入读者印象。建议统一改为 intervention benefit--negative-transfer tradeoff、risk-controlled release 和 bounded released-action non-harm。",
                "Introduction framework 段、Results budget 小节、Discussion",
            ),
            (
                "证据角色需要更早可视化",
                "读者在 Methods 前段需要同时记忆 development、locked external、post-access exploratory、pre-signal robustness 和 prospectively reserved family。建议在主文第一页方法概览中加入紧凑 evidence-role timeline，并在所有表格使用一致的 role 标签。",
                "Methods Study endpoint and evidence roles、Figure framework、Table actions",
            ),
        ],
        "minor": [
            "修正 Methods 中“minimum-budget condition; budget.”残句。",
            "删除多处重复的边界声明，保留 Abstract、每个关键结果末句和 Limitations 中的集中限定。",
            "CRediT 占位符必须在投稿前由全体作者确认。",
        ],
        "confidence": "高",
    },
]


CONSENSUS = [
    ("Cc.1", "缺少成功前瞻外部释放", "审稿人 1、2、3、5", "external validity", "这是从开发域工程证据升级到一区级跨域结论的决定性缺口。"),
    ("Cc.2", "统计独立单位与重复 audit 解释", "审稿人 2、3、4", "statistical rigor", "配置行和 audit repetitions 不能替代独立域；所有推断必须与固定开发数据条件绑定。"),
    ("Cc.3", "创新性必须以完整能力差异呈现", "审稿人 1、3、5", "novelty significance", "仅列出多个组件不足以证明原创性，需要 closest-method taxonomy 和端到端能力比较。"),
    ("Cc.4", "安全与独立性措辞需降级", "审稿人 1、2、5", "claim moderation", "现有证据支持受限 non-harm 和 separately constructed witness，不支持独立验证或普遍安全。"),
    ("Cc.5", "稿件和复现入口需要压缩整合", "审稿人 3、4、5", "reproducibility", "复杂证据链应通过统一数据表、证据角色图和单命令复现入口降低审稿成本。"),
]


TASKS = [
    ("T1", "R1/R3/R5", "完成 AMIGOS 或 Emognition 冻结 one-shot，并保留全部结果", "按既有 lock 执行，不改阈值", "TODO_EXPERIMENT", "授权数据", "外部 gate、manifest、aggregate tables", "Yes"),
    ("T2", "R1", "补充 closest-method taxonomy", "文献与能力矩阵", "TODO_TEXT", "目标期刊文献核验", "Introduction 表和贡献重写", "Yes"),
    ("T3", "R1", "补全 OPCT 优化方程", "加入 soft-target MSE 与正则项", "TODO_TEXT", "无需新数据", "Methods 方程和算法步骤", "Yes"),
    ("T4", "R1/R5", "替换 independent witness 和 safety--efficacy 措辞", "全稿 claim audit", "TODO_TEXT", "无需新数据", "受限术语表和一致措辞", "Yes"),
    ("T5", "R2", "增加更高层 cluster 敏感性", "dataset-task-seed 或 split-family 聚合", "TODO_ANALYSIS", "现有输出", "保守 CI 与结论边界", "Yes"),
    ("T6", "R2", "重写 audit repetitions 的统计解释", "降级为 conditional design sensitivity", "TODO_TEXT", "现有结果", "Methods/Results 限定", "Yes"),
    ("T7", "R2", "将 post-hoc 20% 单元移至补充材料", "主文保留效应量和范围", "TODO_TEXT", "无需新数据", "精简主文段落", "No"),
    ("T8", "R3", "建立 endpoint harmonization 表", "逐域列出构念与标签规则", "TODO_TEXT", "数据协议", "主文或补充表", "Yes"),
    ("T9", "R3", "补齐 EPPVR 伦理和采集信息", "报告审批、同意、基线和设备", "TODO_AUTHOR_CONFIRM", "伦理批件与采集记录", "Methods ethics subsection", "Yes"),
    ("T10", "R4", "建立单一复现入口", "驱动脚本与环境锁", "TODO_ANALYSIS", "现有代码", "README、命令、runtime budget", "No"),
    ("T11", "R4", "自动同步最新验证计数", "从 JSON 报告生成文本", "TODO_ANALYSIS", "现有报告", "无手工漂移的 preview", "No"),
    ("T12", "R5", "压缩 36 页主文", "诊断和失败历史迁移至补充材料", "TODO_TEXT", "无需新数据", "更集中的主文结构", "Yes"),
    ("T13", "R5", "修复残句和 CRediT 占位符", "编辑校对与作者确认", "TODO_AUTHOR_CONFIRM", "作者贡献确认", "投稿清洁稿", "Yes"),
]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def add_body(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.15


def add_concern(doc: Document, number: int, heading: str, body: str, evidence: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.add_run(f"{number}. {heading}。").bold = True
    paragraph.add_run(f"{body} 证据位置：{evidence}。")


def markdown_text() -> str:
    lines = [
        "# DCS OPCT v11 高水平期刊多视角审稿报告",
        "",
        "## 编辑信 Major revision",
        "",
        *EDITOR_PARAGRAPHS,
        "",
    ]
    for reviewer in REVIEWERS:
        lines.extend([f"## {reviewer['role']}", "", "### 整体点评", "", reviewer["overall"], "", "### 主要问题", ""])
        for index, (heading, body, evidence) in enumerate(reviewer["major"], 1):
            lines.extend([f"{index}. **{heading}**。{body} 证据位置：{evidence}。", ""])
        lines.extend(["### 次要问题", ""])
        for index, concern in enumerate(reviewer["minor"], 1):
            lines.append(f"{index}. {concern}")
        lines.extend(["", f"**置信度：{reviewer['confidence']}**", ""])
    lines.extend(["## 跨审稿意见汇总", ""])
    for cid, desc, raised, axis, rationale in CONSENSUS:
        lines.extend([f"**{cid} {desc}** | 提出者：{raised} | 轴：{axis} | 严重性：Major", "", rationale, ""])
    lines.extend([
        "## 修订任务表",
        "",
        "| ID | 审稿人 | 问题 | 策略 | 状态 | 所需输入 | 预期输出 | 是否阻塞 |",
        "|---|---|---|---|---|---|---|---|",
    ])
    lines.extend("| " + " | ".join(row) + " |" for row in TASKS)
    lines.append("")
    return "\n".join(lines)


def build_docx() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.start_type = WD_SECTION.NEW_PAGE

    styles = doc.styles
    for style_name, size in (("Normal", 10.5), ("Title", 16), ("Heading 1", 13), ("Heading 2", 11)):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    title = doc.add_paragraph("DCS OPCT v11 高水平期刊多视角审稿报告", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("编辑信 Major revision", style="Heading 1")
    for paragraph in EDITOR_PARAGRAPHS:
        add_body(doc, paragraph)

    for reviewer in REVIEWERS:
        doc.add_paragraph(reviewer["role"], style="Heading 1")
        doc.add_paragraph("整体点评", style="Heading 2")
        add_body(doc, reviewer["overall"])
        doc.add_paragraph("主要问题", style="Heading 2")
        for index, concern in enumerate(reviewer["major"], 1):
            add_concern(doc, index, *concern)
        doc.add_paragraph("次要问题", style="Heading 2")
        for index, concern in enumerate(reviewer["minor"], 1):
            add_body(doc, f"{index}. {concern}")
        confidence = doc.add_paragraph()
        confidence.add_run(f"置信度：{reviewer['confidence']}").bold = True

    doc.add_paragraph("跨审稿意见汇总", style="Heading 1")
    for cid, desc, raised, axis, rationale in CONSENSUS:
        paragraph = doc.add_paragraph()
        paragraph.add_run(f"{cid} {desc}").bold = True
        paragraph.add_run(f" | 提出者：{raised} | 轴：{axis} | 严重性：Major")
        add_body(doc, rationale)

    doc.add_paragraph("修订任务表", style="Heading 1")
    headers = ("ID", "审稿人", "问题", "策略", "状态", "所需输入", "预期输出", "阻塞")
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    header = table.rows[0]
    set_repeat_table_header(header)
    prevent_row_split(header)
    for index, label in enumerate(headers):
        cell = header.cells[index]
        cell.text = label
        set_cell_shading(cell, "1F4E78")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(8)
    for row_index, values in enumerate(TASKS, 1):
        row = table.add_row()
        prevent_row_split(row)
        cells = row.cells
        for index, value in enumerate(values):
            cells[index].text = value
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2 == 0:
                set_cell_shading(cells[index], "EAF2F8")
            for paragraph in cells[index].paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(7.5)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("nature-review-studio | 2026-09-10 | dcs-opct-v11")
    for run in footer.runs:
        run.font.size = Pt(9)
        run.italic = True

    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MARKDOWN.write_text(markdown_text(), encoding="utf-8")
    build_docx()
    print(DOCX)
    print(MARKDOWN)


if __name__ == "__main__":
    main()
