"""
记忆提取模块 —— 用 LLM 从对话中提炼关键记忆
=============================================
每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得记住的信息，存到数据库里。

v2.3 改进：提取时注入已有记忆，让模型对比后只提取全新信息。
"""

import os
import json
import httpx
from typing import List, Dict

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 记忆模型专用 API Key（不设则回退到主 API_KEY）
# 适用于中转站按模型分组、不同模型需要不同 Key 的场景
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

# 用来提取记忆的模型（便宜的就行）
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "anthropic/claude-haiku-4")

def get_memory_api_key() -> str:
    return MEMORY_API_KEY or API_KEY


EXTRACTION_PROMPT = """你是信息提取专家，负责从对话中识别并提取值得长期记住的关键信息。

# 提取重点
- 关键信息：提取用户的重要信息和值得回忆的生活细节
- 重要事件：记忆深刻的互动，需包含人物、时间、地点（如有）

# 提取范围
- 个人：年龄、生日、职业、学历、居住地
- 偏好：明确表达的喜好或厌恶
- 健康：身体状况、过敏史、饮食禁忌
- 事件：与AI的重要互动、约定、里程碑
- 关系：家人、朋友、重要同事
- 价值观：表达的信念或长期目标
- 情感：重要的情感时刻或关系里程碑
- 生活：用户当天的活动、饮食、出行、日常经历等生活细节
- AI：AI做出的承诺、约定、重要情感表达

# 提取要求
- 事件类记忆保留双方的关键原话，用引号标注是谁说的
- 项目/技术进展只记要点（改了什么、解决了什么），不记调试过程

# 不要提取
- 日常寒暄（"你好""在吗"）
- AI的纯知识性回答（百科、翻译、代码讲解等，不涉及双方关系和承诺的内容）
- 关于记忆系统本身的讨论（"某条记忆没有被记录""记忆遗漏""没有被提取"等）
- AI的思考过程、思维链内容

# 已知信息处理【最重要】
<已知信息>
{existing_memories}
</已知信息>

- 新信息必须与已知信息逐条比对
- 相同、相似或语义重复的信息必须忽略（例如已知"用户去妈妈家吃团年饭"，就不要再提取"用户春节去了妈妈家"）
- 已知信息的补充或更新可以提取（例如已知"用户养了一只猫"，新信息"猫最近生病了"可以提取）
- 与已知信息矛盾的新信息可以提取（标注为更新）
- 仅提取完全新增且不与已知信息重复的内容
- 如果对话中没有任何新信息，返回空数组 []

# 输出格式
请用以下 JSON 格式返回（不要包含其他内容）：
[
  {{"content": "记忆内容", "importance": 分数}},
  {{"content": "记忆内容", "importance": 分数}}
]

importance 分数 1-10，10 最重要。
如果没有值得记住的新信息，返回空数组：[]
"""


async def extract_memories(messages: List[Dict[str, str]], existing_memories: List[str] = None) -> List[Dict]:
    """
    从对话消息中提取记忆

    参数：
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}, ...]
        existing_memories: 已有记忆内容列表，用于去重对比

    返回：
        记忆列表，格式 [{"content": "...", "importance": N}, ...]
    """
    if not API_KEY:
        print("⚠️  API_KEY 未设置，跳过记忆提取")
        return []

    if not messages:
        return []

    # 把对话格式化成文本
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"用户: {content}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content}\n"

    if not conversation_text.strip():
        return []

    # 格式化已有记忆
    if existing_memories:
        memories_text = "\n".join(f"- {m}" for m in existing_memories)
    else:
        memories_text = "（暂无已知信息）"

    # 把已有记忆填入prompt
    prompt = EXTRACTION_PROMPT.format(existing_memories=memories_text)

    # 调用 LLM 提取记忆
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {get_memory_api_key()}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://midsummer-gateway.local",
                    "X-Title": "Midsummer Memory Extraction",
                },
                json={
                    "model": MEMORY_MODEL,
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"请从以下对话中提取新的记忆：\n\n{conversation_text}"},
                    ],
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆提取请求失败: {response.status_code}")
                return []

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 打印模型原始返回（截断防刷屏）
            print(f"📝 记忆模型原始返回:\n{text[:500]}", flush=True)

            # 清理可能的 markdown 格式
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # 强力JSON提取：如果上面清理后仍然解析失败，用正则兜底
            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                # 尝试从文本中提取第一个 [...] 结构
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                        print(f"📝 JSON正则兜底提取成功")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  记忆提取结果解析失败: {e}")
                        return []
                else:
                    print(f"⚠️  记忆提取结果中未找到JSON数组")
                    return []

            if not isinstance(memories, list):
                return []

            # 验证格式
            valid_memories = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    valid_memories.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                    })

            print(f"📝 从对话中提取了 {len(valid_memories)} 条新记忆（已对比 {len(existing_memories or [])} 条已有记忆）")
            return valid_memories

    except json.JSONDecodeError as e:
        print(f"⚠️  记忆提取结果解析失败: {e}")
        return []
    except Exception as e:
        print(f"⚠️  记忆提取出错: {e}")
        return []


SCORING_PROMPT = """你是记忆重要性评分专家。请对以下记忆条目逐条评分。

# 评分规则（1-10）
- 9-10：核心身份信息（名字、生日、职业、重要关系）
- 7-8：重要偏好、重大事件、深层情感
- 5-6：日常习惯、一般偏好
- 3-4：临时状态、偶然提及
- 1-2：琐碎信息

# 输入记忆
{memories_text}

# 输出格式
返回 JSON 数组，每条包含原文和评分：
[{{"content": "原文", "importance": 评分数字}}]

只返回 JSON，不要其他文字。"""


async def score_memories(texts: List[str]) -> List[Dict]:
    """对纯文本记忆条目批量评分"""
    if not texts:
        return []

    memories_text = "\n".join(f"- {t}" for t in texts)
    prompt = SCORING_PROMPT.format(memories_text=memories_text)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {get_memory_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MEMORY_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 4000,
                },
            )

            if response.status_code != 200:
                print(f"⚠️  记忆评分请求失败: {response.status_code}")
                # 失败时返回默认分数
                return [{"content": t, "importance": 5} for t in texts]

            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                memories = json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        memories = json.loads(match.group())
                    except json.JSONDecodeError:
                        return [{"content": t, "importance": 5} for t in texts]
                else:
                    return [{"content": t, "importance": 5} for t in texts]

            if not isinstance(memories, list):
                return [{"content": t, "importance": 5} for t in texts]

            valid = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    valid.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                    })

            print(f"📝 为 {len(valid)} 条记忆完成自动评分")
            return valid

    except Exception as e:
        print(f"⚠️  记忆评分出错: {e}")
        return [{"content": t, "importance": 5} for t in texts]
