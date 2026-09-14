from __future__ import annotations

from pydantic import BaseModel, Field


class LabItemInput(BaseModel):
    item_name: str = Field(min_length=1, max_length=64)
    test_value: str = Field(min_length=1, max_length=32)
    unit: str | None = Field(default=None, max_length=32)


class LabReportInterpretRequest(BaseModel):
    lab_item_list: list[LabItemInput] = Field(min_length=1)
    sync_to_archive: bool = False


class LabItemOutput(BaseModel):
    item_name: str
    test_value: str
    reference_range: str | None = None
    abnormal_flag: str | None = None
    #: 这个状态是谁给的：`image_flag`（化验单上标注的）/ `image_blank`（图上没标，
    #: 按"只标异常项"的规范视为正常）/ `general_range`（无图，按库内参考区间算的）。
    #: 用户有权知道哪句话是医院给的结论、哪句是系统按通用口径推的。
    judge_source: str | None = None
    #: 「偏高/偏低之后怎么办」的结构化文本：可能原因 / 建议 / 何时就医 / 说明。
    #: 只有库内有对应条目、且状态为 H / L 时才存在。
    advice: dict | None = None
    meaning: str


class LabReportInterpretResponse(BaseModel):
    item_list: list[LabItemOutput]
    final_desc: str


# --------------------------------------------------------------------------
# 视觉抽取契约（化验单图像识别）
# --------------------------------------------------------------------------


class LabVisionItem(BaseModel):
    """视觉模型对**单个检验项**的输出契约。

    字段名与 `lab_items`（`item_name` / `test_value` / `unit`）刻意保持一致：
    识别结果最终要回流成同一种结构，字段名不一致就得在两处做映射。

    `reference_range` 是**从图上抄下来的**，不是查库得到的 —— 库里那一份由
    `LabReferenceService` 提供，两者口径不同（前者是这家医院这张单子印的，
    后者是通用口径），所以分开存放，不互相覆盖。

    全部字段给默认空串：模型漏字段时退化成"该项信息不全"，而不是整次调用
    校验失败。`strictify()` 会把它们全部塞进 `required`，所以正常情况下
    模型仍会逐字段输出；默认值只是客户端侧的**兜底**。
    """

    item_name: str = Field(default="", max_length=64)
    test_value: str = Field(default="", max_length=32)
    unit: str = Field(default="", max_length=32)
    reference_range: str = Field(default="", max_length=64)
    #: 结果列上的异常标记：`H` / `L` / 空串。
    #:
    #: **与 `test_value` 分开存是为了不丢信息**：整串 `6.5↑` 填进 `test_value`
    #: 会让数值侧的形态校验失败（`_VALUE_RE` 只认纯数字），这一条就整个进不了
    #: 判定；拆成两个字段后，数值照样能参与比大小，标记也能单独承载。
    #:
    #: 采信这个标记 = 采信**这家医院按该患者**给出的判定方向，口径比通用参考区间
    #: 更准（医院只给异常项打标记，所以空白即正常）。
    #: 模型可能不照要求填或填了非法值，一律在 `_normalize_flag` 里校验后才采信。
    abnormal_flag: str = Field(default="", max_length=8)


class LabVisionReport(BaseModel):
    """一张化验单图的整体输出。

    `items` **允许为空**：拍到无关图片（或画面里确实没有检验项目）时，
    正确答案就是空数组，不该报校验错误 —— 那会把"没识别到"误报成"调用失败"。
    """

    items: list[LabVisionItem] = Field(default_factory=list, max_length=60)


class LabImageExtractRequest(BaseModel):
    """图片识别的入参。

    **为什么是 base64 JSON 而不是 multipart 上传**：

    1. multipart 要求 `python-multipart`，项目当前未安装也未声明 ——
       FastAPI 在**定义路由时**就会因缺它而抛错，等于把整个应用的导入炸掉
       （`app.main` 导入期失败）。为一个图片入口引入这种全局风险不值得。
    2. base64 正是送模型的形态（`data:` URL），前端 `FileReader.readAsDataURL`
       直接给的就是它，转成 multipart 反而多一层解析。
    3. 与既有接口统一为 JSON + `APIResponse`，前端复用同一个请求封装。

    代价是请求体大约多 33%（base64 膨胀），由 `LAB_IMAGE_MAX_MB` 在解码**之前**
    就按编码长度挡掉，不会先吃满内存再报错。
    """

    image_base64: str = Field(min_length=32, description="图片 base64；允许带 `data:image/png;base64,` 前缀")
    mime: str = Field(default="", max_length=64, description="可选；带 data 前缀时以前缀为准")


class LabImageItemOut(BaseModel):
    """回填前每个条目的**完整去向**，供前端逐条展示与排错。"""

    raw_name: str
    item_name: str
    test_value: str
    unit: str
    reference_range: str
    #: 从图上读到的异常标记（`H` / `L` / 空串）。
    #: 这是**识别结果**不是判定结论，所以放在这里；判定仍由
    #: `LabReportTool.interpret` 统一做（图路与文路共用同一个判定实现）。
    image_flag: str = ""
    in_reference_base: bool
    lib_unit: str | None = None
    unit_status: str = "unknown"
    fillable: bool = True
    note: str = ""


class LabImageExtractResponse(BaseModel):
    """图片识别响应：**只给回填材料，不给解读结论**。

    刻意不含 `meaning` / `abnormal_flag`（判定结论）—— 解读由用户在文本框里确认后
    走原有的 `/lab/report-interpret`（或直接发消息），
    保证"图路"与"文路"最终经过**同一个**判定实现。
    条目上的 `image_flag` 不违背这一点：它是**从图上读出来的原始标记**，
    不是我们算出来的结论。
    """

    item_count: int
    items: list[LabImageItemOut]
    fill_text: str
    uncovered: list[str]
    unit_mismatch: list[dict]
    dropped: list[dict]
    warnings: list[str]
    image_meta: dict = Field(default_factory=dict)
