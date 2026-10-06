#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import locust.stats
from locust import HttpUser, between, task

locust.stats.CSV_STATS_INTERVAL_SEC = 5  # default is paymentservice_timeout second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5  # Determines how often the data is flushed to disk, default is 10 seconds

from locust import HttpUser, task, between
import random
import os

# ==========================================================================
# 故障感知的权重配置
# ==========================================================================
# 当前注入故障的服务，由 chaos_service.sh 通过环境变量传入(缺省 pdf-parsing)。
# locust 会据此把请求分布向"能调用到该故障服务"的任务倾斜，从而在故障窗口
# 内产生更多经过故障服务的调用链，方便故障定位/根因分析。
FAULT_SERVICE = os.getenv("FAULT_SERVICE", "agent-network-pdf-parsing")

# 权重基数（值越大被抽中的概率越高）
W_FAULT_COMPLEX = 6   # 命中故障服务 且 复杂多跳链路
W_FAULT_SIMPLE = 3     # 命中故障服务 且 简单任务
W_OTHER_COMPLEX = 2    # 未命中故障服务 的复杂任务
W_OTHER_SIMPLE = 1     # 未命中故障服务 的简单任务


def task_services(task_def):
    """任务实际经过的服务调用链。

    显式声明的 `services` 优先；未声明时退化为任务自身的 `name`
    （对单服务的 SIMPLE 任务而言这就是它调用的唯一服务）。
    """
    return task_def.get("services", [task_def["name"]])


def task_weight(task_def):
    """根据是否命中当前故障服务、以及任务复杂度计算抽样权重。"""
    hits_fault = FAULT_SERVICE in task_services(task_def)
    is_complex = task_def.get("type") == "COMPLEX"
    if hits_fault and is_complex:
        return W_FAULT_COMPLEX
    if hits_fault:
        return W_FAULT_SIMPLE
    if is_complex:
        return W_OTHER_COMPLEX
    return W_OTHER_SIMPLE


class MDOCUser(HttpUser):
    wait_time = between(5, 10)

    # ==========================================
    # 1. 国内可访问的公开文件资源池
    # ==========================================
    FILE_RESOURCES = {
        # ============================================================
        # PDF文件 - 很短的中性 PDF（正文很短，降低 token 消耗）
        # 注意1: 下面几篇较长的中性 PDF 已注释保留，后续需要时自行取消注释改回。
        # 注意2: 原政府公报PDF仍保留在服务器上，登记为 fault_injection.py 的
        #        "content_compliance"（不符合法律法规）故障用例，请勿删除。
        # ============================================================
        "pdf": [
            # 短文档: PDF 测试文件 (~57词, orimi.com pdf-test)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfcGRmdGVzdC5wZGY&user=exp",

            # 短文档: PDF sample (~173词, unec.edu.az pdf-sample)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfc2FtcGxlLnBkZg&user=exp",

            # 短文档: XSL-FO basic-link 样例 (~145词, antennahouse)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c2hvcnRfbGluay5wZGY&user=exp",

            # ---- 以下为较长的中性 PDF，暂时注释；需要时取消注释改回 ----
            # Attention Is All You Need (arXiv:1706.03762)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=YXR0ZW50aW9uX2lzX2FsbF95b3VfbmVlZC5wZGY&user=exp",
            # BERT 预训练论文 (arXiv:1810.04805)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=YmVydF9wcmV0cmFpbmluZ19wYXBlci5wZGY&user=exp",
            # 体感神经科学教材样例 (css4.pub somatosensory)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=c29tYXRvc2Vuc29yeV90ZXh0Ym9vay5wZGY&user=exp",
            # USENIX 会议论文样例 (css4.pub usenix example)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=dXNlbml4X2V4YW1wbGVfcGFwZXIucGRm&user=exp",
            # Drylab 科普通讯样例 (css4.pub drylab newsletter)
            # "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZHJ5bGFiX25ld3NsZXR0ZXIucGRm&user=exp",
        ],

        # ============================================================
        # Excel文件 - 中国大陆公开数据
        # ============================================================
        "excel": [
            # 市场监管总局：糕点监督抽检不合格产品信息 (原: samr.gov.cn 090ad9e0...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMC54bHN4&user=exp",

            # 市场监管总局：蔬菜制品监督抽检不合格产品信息 (原: samr.gov.cn 0595a450...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMS54bHN4&user=exp",

            # 市场监管总局：水果制品监督抽检不合格产品信息 (原: samr.gov.cn f8ceb0d7...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMi54bHN4&user=exp",

            # 市场监管总局：方便食品监督抽检不合格产品信息 (原: samr.gov.cn bc79e9e8...xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfMy54bHN4&user=exp",

            # 中央财经大学：学籍档案远程查档申请表 (原: archives.cufe.edu.cn ...2F76.xlsx)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=ZXhjZWxfNC54bHN4&user=exp",
        ],

        # ============================================================
        # Word文档 - 中国大陆公开文档
        # ============================================================
        "word": [
            # 深圳市龙岗区耳鼻咽喉医院：报名表
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=MTIzNTE5MzUuZG9jeA&user=exp",
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=5Lit5Zu96ZO26KGMMjAyNuW5tOagoeWbreWcsOWbveWutuWKqeWtpui0t-asvuWtpueUn-WcqOe6v-eUs-ivt-aMh-WNly5kb2N4&user=exp",
            # 市场监管总局：部分不合格检验项目小知识
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=6ZmE5Lu2MSvpg6jliIbkuI3lkIjmoLzmo4Dpqozpobnnm67lsI_nn6Xor4YuZG9jeA&user=exp",
        ],

        # ============================================================
        # 图片 - 简单图片 / OCR测试
        # ============================================================
        "image_ocr": [
            # 国务院：澳门特别行政区行政区域图 (原: https://www.gov.cn/zhengce/aomen.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzAuanBlZw&user=exp",

            # 国家中医药管理局规划相关图片 (原: ...W020230506781423064462.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzEuanBlZw&user=exp",

            # 国家中医药管理局规划相关图片 (原: ...W020230506781423164720.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzIuanBlZw&user=exp",

            # 国家中医药管理局规划相关图片 (原: ...W020230506781423260634.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzMuanBlZw&user=exp",

            # 国家中医药管理局规划相关图片 (原: ...W020230506781423338851.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2Vfb2NyXzQuanBlZw&user=exp",

            # 环境与智能体感知架构图
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=Q2hhdEdQVCBJbWFnZSAyMDI25bm0N-aciDE15pelIDE3XzM0XzQwLnBuZw&user=exp"
        ],

        # ============================================================
        # 复杂图片 - 图表 / PNG / JPG等
        # ============================================================
        "image_complex": [
            # 国家中医药管理局规划相关PNG (原: ...W020230506781422832145.png)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8wLnBuZw&user=exp",

            # 国家中医药管理局规划相关JPG (原: ...W020230506781422944424.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8xLmpwZWc&user=exp",

            # 国家中医药管理局规划相关JPG (原: ...W020230506781423064462.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8yLmpwZWc&user=exp",

            # 国家中医药管理局规划相关JPG (原: ...W020230506781423164720.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF8zLmpwZWc&user=exp",

            # 国家中医药管理局规划相关JPG (原: ...W020230506781423260634.jpg)
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=aW1hZ2VfY29tcGxleF80LmpwZWc&user=exp",
            # 山水画
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=T0RGRk1rWkNOemRCTjBJd05FRXlPRGxHTlRjd056STBPVGcxTWpoQk5VWGt1SzNsbTczbHNiSG1zTFRubEx2cHU0VGxzYkhrdXBIbXRiZm1sNlhsaDdveU1ESTJNRGt3Tmkwd09ERTJOVGd1Y0c1bi5wbmc&user=exp",
            # 微服务架构图
            "http://192.168.31.15:12104/api/sys/storage/file/download?fileName=Q2hhdEdQVCBJbWFnZSAyMDI25bm0N-aciDE15pelIDAwXzQzXzQwLnBuZw&user=exp",
        ],
    }

    # ==========================================
    # 2. 任务配置列表 - 大幅扩充
    # ==========================================
    TASK_DEFINITIONS = [
        # ==================== 简单任务 ====================
        # CSV生成类
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "生成一份包含姓名、年龄、部门、工资的CSV格式员工数据，共20条。"
        },
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "生成一份学生成绩表CSV，包含学号、姓名、语文、数学、英语成绩，共30条记录。"
        },
        {
            "name": "agent-network-csv-gen",
            "type": "SIMPLE",
            "prompt": "生成一份产品销售数据CSV，包含产品ID、名称、类别、单价、销量，共50条。"
        },

        # 路线规划类
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "规划从北京首都机场到市中心的驾车路线，避开拥堵。"
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "规划从上海虹桥火车站到外滩的公共交通路线。"
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "规划从广州塔到白云机场的地铁路线，包含换乘信息。"
        },
        {
            "name": "agent-network-direction",
            "type": "SIMPLE",
            "prompt": "规划深圳市内从南山科技园到宝安机场的最快路线。"
        },

        # Excel生成类
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "生成一份2024年第一季度的销售预测Excel表格，包含月份、产品类别、预估销售额、实际销售额。"
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "生成一份员工考勤统计表Excel，包含姓名、部门、出勤天数、请假天数、加班时长。"
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "生成一份家庭月度预算表Excel，包含收入、支出分类、预算金额、实际金额。"
        },
        {
            "name": "agent-network-excel-gen",
            "type": "SIMPLE",
            "prompt": "生成一份库存管理表Excel，包含商品编码、名称、库存量、安全库存、供应商。"
        },

        # 图片生成类
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "生成一张'赛博朋克风格的未来城市，夜晚，霓虹灯'的高清图片。"
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "生成一张'中国山水画风格，黄山云海，日出'的图片。"
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "生成一张'现代简约风格的客厅设计，白色为主色调'的室内设计图。"
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "生成一张'可爱的卡通猫咪，水彩画风格'的插画。"
        },
        {
            "name": "agent-network-image-gen",
            "type": "SIMPLE",
            "prompt": "生成一张'科技感的AI机器人，蓝色调，未来感'的概念图。"
        },

        # 旅游规划类
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "制定一份详细的'云南大理3天2夜'旅游攻略，包含住宿建议、景点推荐、美食推荐。"
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "制定一份'成都5天4夜'亲子游行程，包含大熊猫基地、都江堰等景点。"
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "制定一份'西安历史文化3日游'计划，包含兵马俑、华清池、古城墙等。"
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "制定一份'杭州西湖周边2日游'攻略，包含灵隐寺、龙井村等景点。"
        # },
        # {
        #     "name": "agent-network-planner",
        #     "type": "SIMPLE",
        #     "prompt": "制定一份'三亚海滨度假4天3夜'行程，包含海滩、海鲜、水上活动。"
        # },

        # 文本总结类
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "总结以下文本的核心观点：'人工智能正在改变世界，大模型技术使得自然语言处理变得更加高效。深度学习算法在图像识别、语音识别等领域取得了突破性进展。'"
        # },
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "总结这篇新闻报道的主要内容：'2024年中国经济稳步增长，GDP同比增长5.2%，消费市场持续复苏，高新技术产业快速发展。'"
        # },
        # {
        #     "name": "agent-network-summarizer",
        #     "type": "SIMPLE",
        #     "prompt": "提取以下会议纪要的关键决策点：'会议讨论了新产品上线计划、市场推广策略、人员配置调整等议题。'"
        # },

        # Word文档生成类
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "生成一份'项目立项申请书'的Word文档大纲，包含项目背景、目标、实施方案、预算。"
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "生成一份'年度工作总结报告'的Word文档模板，包含工作回顾、成果展示、问题分析、下年计划。"
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "生成一份'商业计划书'的Word文档结构，包含市场分析、商业模式、财务预测。"
        },
        {
            "name": "agent-network-word-gen",
            "type": "SIMPLE",
            "prompt": "生成一份'员工培训方案'的Word文档，包含培训目标、课程内容、时间安排、考核方式。"
        },

        # ==================== 文件解析类任务 ====================
        # PDF解析
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "请解析这个PDF文件，提取文档标题、主要章节和关键信息：{url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "分析这个PDF文件的内容结构，列出所有表格并提取数据：{url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "从这个PDF文件中提取所有的日期、数字和关键指标：{url}"
        },
        {
            "name": "agent-network-pdf-parsing",
            "type": "SIMPLE",
            "requires_file": "pdf",
            "prompt": "解析PDF文件，总结文档的核心内容和主要结论：{url}"
        },

        # Excel解析
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "读取这个Excel文件，列出所有工作表名称和每个表的列名：{url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "分析这个Excel文件，计算数值列的总和、平均值、最大值和最小值：{url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "从这个Excel文件中提取前10行数据，并转换为JSON格式：{url}"
        },
        {
            "name": "agent-network-excel-parsing",
            "type": "SIMPLE",
            "requires_file": "excel",
            "prompt": "分析Excel文件中的数据趋势，找出异常值和规律：{url}"
        },

        # Word解析
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "解析这个Word文档，提取所有标题和段落内容：{url}"
        },
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "从这个Word文档中提取所有的表格数据：{url}"
        },
        {
            "name": "agent-network-word-parsing",
            "type": "SIMPLE",
            "requires_file": "word",
            "prompt": "分析Word文档的结构，提取目录和章节信息：{url}"
        },

        # OCR识别
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "识别这张图片中的所有文字内容，包括中文和英文：{url}"
        },
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "提取图片中的数字、日期和关键信息：{url}"
        },
        {
            "name": "agent-network-ocr",
            "type": "SIMPLE",
            "requires_file": "image_ocr",
            "prompt": "识别图片中的文字，并转换为可编辑的文本格式：{url}"
        },

        # 图片分析
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "详细描述这张图片的内容，包括颜色、物体、场景和布局：{url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "分析这张图片的风格、主题和视觉元素：{url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "识别图片中的品牌logo、文字和主要元素：{url}"
        },
        {
            "name": "agent-network-image",
            "type": "SIMPLE",
            "requires_file": "image_complex",
            "prompt": "评估这张图片的质量、构图和色彩搭配：{url}"
        },

        # ==================== 组合型/长链路任务 ====================
        # 图片+报告生成
        {
            "name": "agent-network-combo-image-report",
            "type": "COMPLEX",
            "services": ["agent-network-image", "agent-network-word-gen"],
            "requires_file": "image_complex",
            "prompt": "请先分析这张图片的内容（{url}），然后基于图片内容生成一份详细的'视觉分析报告'，包含场景描述、元素分析、色彩分析和建议。"
        },
        {
            "name": "agent-network-combo-ocr-summary",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "先识别这张图片中的文字（{url}），然后对识别出的内容进行总结和分类，生成一份结构清晰的PDF摘要报告。"
        },
        {
            "name": "agent-network-combo-ocr-pdf-transcript",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "第一步：对图片（{url}）进行OCR识别，完整提取其中的中英文文字、数字和日期；"
                      "第二步：校正明显的识别错误并保留原有段落层次；"
                      "第三步：将整理后的内容生成一份可归档的PDF文字转录文档。"
        },
        {
            "name": "agent-network-combo-ocr-pdf-key-info",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "第一步：识别图片（{url}）中的全部文字；"
                      "第二步：提取标题、日期、地点、机构名称、关键数字等重要信息；"
                      "第三步：生成一份包含原文摘录和关键信息表格的PDF报告。"
        },
        {
            "name": "agent-network-combo-ocr-pdf-bilingual",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "先对图片（{url}）执行OCR，分别识别中文和英文内容；"
                      "再按语言分类整理识别结果，并生成一份包含识别原文、内容概述和分类结果的PDF文档。"
        },
        {
            "name": "agent-network-combo-ocr-pdf-audit",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-pdf-gen"],
            "requires_file": "image_ocr",
            "prompt": "对图片（{url}）进行OCR识别，逐项核对并标注疑似识别错误、模糊字符和缺失内容，"
                      "最后生成一份包含识别结果、问题清单和置信度说明的PDF质检报告。"
        },
        
        # Excel+数据分析
        {
            "name": "agent-network-combo-excel-analysis",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-word-gen"],
            "requires_file": "excel",
            "prompt": "读取这个Excel文件（{url}），分析数据趋势，生成一份'数据分析报告'，包含关键发现、图表建议和改进建议。"
        },
        {
            "name": "agent-network-combo-excel-visualization",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen"],
            "requires_file": "excel",
            "prompt": "分析这个Excel文件的数据（{url}），然后生成一份包含数据可视化的'业务洞察报告'。"
        },
        
        # PDF+总结
        {
            "name": "agent-network-combo-pdf-report",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "解析这个PDF文件（{url}），提取关键信息，然后生成一份'文档摘要报告'，包含背景、主要内容、结论和建议。"
        },
        {
            "name": "agent-network-combo-pdf-excel",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen"],
            "requires_file": "pdf",
            "prompt": "从PDF文件（{url}）中提取所有表格数据，并整理成结构化的Excel格式描述。"
        },
        
        # 多步骤规划任务
        {
            "name": "agent-network-combo-travel-budget",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-excel-gen"],
            "prompt": "我需要去新疆伊犁旅游7天，请先生成详细行程规划，然后基于行程生成一份'预算估算表'，包含交通、住宿、餐饮、门票等费用。"
        },
        {
            "name": "agent-network-combo-marketing-plan",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-word-gen"],
            "prompt": "为一款新的智能手机制定营销方案，先进行市场分析，然后生成'营销策略文档'，包含目标用户、渠道策略、推广计划。"
        },
        {
            "name": "agent-network-combo-product-launch",
            "type": "COMPLEX",
            "services": ["agent-network-planner", "agent-network-word-gen"],
            "prompt": "规划一个新产品发布会，先制定活动流程，然后生成'执行方案'，包含时间安排、人员分工、物料清单。"
        },
        {
            "name": "agent-network-combo-data-visualization",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen"],
            "prompt": "我有一份销售数据，请分析数据特点，然后生成一份'数据可视化方案'，建议适合的图表类型和展示方式。"
        },
        
        # 文档转换类
        {
            "name": "agent-network-combo-pdf-to-word",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "解析PDF文件（{url}），提取内容并重新组织成Word文档格式，包含标题、段落、列表等结构化元素。"
        },
        {
            "name": "agent-network-combo-word-to-ppt",
            "type": "COMPLEX",
            "services": ["agent-network-word-parsing", "agent-network-word-gen"],
            "requires_file": "word",
            "prompt": "分析Word文档（{url}）的内容，然后生成一份PPT大纲，包含每页的标题、要点和视觉建议。"
        },

        # ==================== 新增：3+ 服务长链路 COMPLEX 任务 ====================
        # 这些任务显式规划 3 个及以上服务调用链路，且大多经过 pdf-parsing 故障服务，
        # 在故障窗口内可产生更丰富的多跳调用链，便于观测故障在链路上的传播。
        {
            "name": "agent-network-combo-pdf-excel-word",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "第一步：解析这个PDF文件（{url}），提取其中的表格与关键数据；"
                      "第二步：把提取到的表格整理成结构化的Excel数据；"
                      "第三步：基于Excel数据生成一份完整的Word分析报告，包含数据概览、趋势分析与结论建议。"
        },
        {
            "name": "agent-network-combo-pdf-excel-chart",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-gen", "agent-network-image-gen"],
            "requires_file": "pdf",
            "prompt": "第一步：解析这个PDF文件（{url}），抽取其中的表格与数值指标；"
                      "第二步：把抽取到的数据整理成结构化的Excel表格；"
                      "第三步：基于该Excel数据生成一张数据可视化图表，直观展示关键指标。"
        },
        {
            "name": "agent-network-combo-pdf-excel-analysis-report",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-excel-parsing",
                         "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "pdf",
            "prompt": "第一步：解析PDF文件（{url}），抽取其中所有表格；"
                      "第二步：把表格作为Excel数据读入并做统计分析（求和/均值/异常值）；"
                      "第三步：将分析结果重新组织成一张新的Excel汇总表；"
                      "第四步：生成一份Word版《数据洞察报告》，说明关键发现与改进建议。"
        },
        {
            "name": "agent-network-combo-ocr-excel-report",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-excel-gen", "agent-network-word-gen"],
            "requires_file": "image_ocr",
            "prompt": "第一步：对这张明确URL指向的图片（{url}）做OCR识别，提取其中的文字、数字和日期；"
                      "第二步：将OCR结果按信息类型整理成结构化的Excel表格；"
                      "第三步：基于识别和整理结果，生成一份Word版《图片文字识别报告》，包含文字摘录、关键信息和分类汇总。"
        },
        {
            "name": "agent-network-combo-image-word",
            "type": "COMPLEX",
            "services": ["agent-network-image", "agent-network-word-gen"],
            "requires_file": "image_complex",
            "prompt": "第一步：分析图片（{url}）的场景、布局、颜色和主要视觉元素；"
                      "第二步：综合视觉分析与识别结果，生成一份结构化的Word图像分析报告。"
        },
        {
            "name": "agent-network-combo-ocr-excel-chart",
            "type": "COMPLEX",
            "services": ["agent-network-ocr", "agent-network-excel-gen", "agent-network-image-gen"],
            "requires_file": "image_ocr",
            "prompt": "第一步：识别图片（{url}）中的文字、数字、日期和表格信息；"
                      "第二步：将可量化信息整理成结构化Excel表格；"
                      "第三步：基于表格中的关键数据生成一张可视化图表。"
        },
        {
            "name": "agent-network-combo-pdf-word-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-pdf-parsing", "agent-network-word-gen", "agent-network-pdf-gen"],
            "requires_file": "pdf",
            "prompt": "第一步：解析PDF文件（{url}）并提取标题、章节和正文；"
                      "第二步：将提取内容重新组织成结构规范的Word文档；"
                      "第三步：把整理后的文档生成一份版式清晰的新PDF文件。"
        },
        {
            "name": "agent-network-combo-excel-chart-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-excel-parsing", "agent-network-image-gen", "agent-network-pdf-gen"],
            "requires_file": "excel",
            "prompt": "第一步：解析Excel文件（{url}）并分析关键指标和数据趋势；"
                      "第二步：根据分析结果生成合适的数据可视化图表；"
                      "第三步：将指标说明和图表排版生成一份PDF数据报告。"
        },
        {
            "name": "agent-network-combo-word-image-pdf",
            "type": "COMPLEX",
            "services": ["agent-network-word-parsing", "agent-network-image-gen", "agent-network-pdf-gen"],
            "requires_file": "word",
            "prompt": "第一步：解析Word文档（{url}），提取标题、章节、重点内容和数据；"
                      "第二步：根据文档重点生成一张配套的信息图；"
                      "第三步：将原文要点和信息图整合生成一份PDF简报。"
        },
    ]

    # 与 TASK_DEFINITIONS 一一对应的抽样权重：命中当前故障服务(FAULT_SERVICE)
    # 的任务权重更高，其中复杂多跳链路又高于简单任务。
    TASK_WEIGHTS = [task_weight(td) for td in TASK_DEFINITIONS]

    @task
    def run_mixed_tasks(self):
        # 按故障感知权重随机选择一个任务定义（向故障服务倾斜）
        task_def = random.choices(self.TASK_DEFINITIONS, weights=self.TASK_WEIGHTS, k=1)[0]

        # 处理文件 URL (如果需要)
        final_prompt = task_def["prompt"]
        if task_def.get("requires_file"):
            file_type = task_def["requires_file"]
            file_urls = self.FILE_RESOURCES.get(file_type, [])
            if file_urls:
                # 随机选择一个文件URL
                file_url = random.choice(file_urls)
                final_prompt = final_prompt.format(url=file_url)
            else:
                print(f"Warning: No URL found for file type {file_type}")

        # 构造请求 Payload
        payload = {
            "flowId": "@cn.com.thingo.intelligentAgentPlatform.taskScheduling/FLOW_OPEN_TASK",
            "params": {
                "openTask": {
                    "userId": "CDA7B6E29E2B4CBBB53E424A68A1B85C",
                    "organizeId": "C122FBF575E94BF8867580A535E20BD3",
                },
                "task": final_prompt
            }
        }

        # 发送请求并标记任务类型
        task_name = task_def["name"]
        task_type = task_def["type"]

        import uuid
        id = uuid.uuid4()
        # 每个请求只打一行，便于关联 id/任务类型/任务名/prompt，同时降低日志噪音
        print(f"id:{id} {task_type}/{task_name} | {final_prompt}")

        with self.client.post(
                "/api/engine/flow",
                json=payload,
                name=f"{task_type}/{task_name}",
                catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code} for {task_name}")
