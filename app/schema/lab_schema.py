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
    in_reference_base: bool
    lib_unit: str | None = None
    unit_status: str = "unknown"
    fillable: bool = True
    note: str = ""


class LabImageExtractResponse(BaseModel):
    """图片识别响应：**只给回填材料，不给解读结论**。

    刻意不含 `abnormal_flag` / `meaning` —— 解读由用户在文本框里确认后
    走原有的 `/lab/report-interpret`（或直接发消息），
    保证"图路"与"文路"最终经过**同一个**判定实现。
    """

    item_count: int
    items: list[LabImageItemOut]
    fill_text: str
    uncovered: list[str]
    unit_mismatch: list[dict]
    dropped: list[dict]
    warnings: list[str]
    image_meta: dict = Field(default_factory=dict)
