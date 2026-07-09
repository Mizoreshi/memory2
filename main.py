"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
让你的 AI 拥有长期记忆。

工作原理：
1. 接收客户端（Kelivo / ChatBox / 任何 OpenAI 兼容客户端）的消息
2. 自动搜索数据库中的相关记忆，注入 system prompt
3. 转发给 LLM API（支持 OpenRouter / OpenAI / 任何兼容接口）
4. 后台自动存储对话 + 用 AI 提取新记忆

环境变量 MEMORY_ENABLED=false 时退化为纯转发网关（第一阶段）。
"""

import os
import json
import uuid
import asyncio
import secrets
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import init_tables, close_pool, save_message, search_memories, save_memory, get_all_memories_count, get_recent_memories, get_all_memories, get_pool, get_all_memories_detail, update_memory, delete_memory, delete_memories_batch, get_gateway_config, set_gateway_config, get_all_gateway_config, get_conversation_messages, get_session_cache_state, save_session_cache_state, delete_session_cache_state, save_token_usage, ensure_token_usage_table, get_conversations_paginated, delete_conversation, batch_delete_conversations, merge_sessions_to_target, list_all_session_cache_states, export_all_conversations, import_conversations, get_last_user_content, update_last_assistant_message, db_row_to_message, backfill_memory_embeddings, get_pending_memory_embedding_count, search_conversations, update_message_content, delete_single_message, rename_session_id, get_fragments_by_date, get_fragments_by_date_range, create_event_memory, deactivate_memories, promote_to_core, merge_memories, check_duplicate_memory, update_memory_with_layer, get_layer_statistics, cleanup_old_fragments, revert_merge
import database as _db_module  # 用于 /api/settings 热更新 database.py 全局变量
from memory_extractor import extract_memories, score_memories

# ============================================================
# 配置项 —— 全部从环境变量读取，部署时在云平台面板里设置
# ============================================================
import os
os.environ["DATABASE_URL"] = "postgresql://bing_dao_shi_ji_user:NPRH3Zxrgjf44IzgNzbmvx668bkyEXfG@dpg-d97gvfm7r5hc73bd2cq0-a.virginia-postgres.render.com/bing_dao_shi_ji"

# 你的 API Key（OpenRouter / OpenAI / 其他兼容服务）
API_KEY = os.getenv("API_KEY", "")

# API 地址（改这个就能切换不同的 LLM 服务商）
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 默认模型（如果客户端没指定就用这个）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")

# 网关端口
PORT = int(os.getenv("PORT", "8080"))

# 网关访问密钥
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")

# 记忆系统开关
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"

# 每次注入的最大记忆条数
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))

# 记忆提取间隔
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))

# 记忆提取+注入总开关
MEMORY_EXTRACT_ENABLED = os.getenv("MEMORY_EXTRACT_ENABLED", "true").lower() == "true"

# 分区缓存
CACHE_PARTITION_ENABLED = os.getenv("CACHE_PARTITION_ENABLED", "false").lower() == "true"
CACHE_PARTITION_X = int(os.getenv("CACHE_PARTITION_X", "15"))
CACHE_SUMMARY_MODEL = os.getenv("CACHE_SUMMARY_MODEL", "")  # 留空=不生成摘要
CACHE_PARTITION_TRIGGER = os.getenv("CACHE_PARTITION_TRIGGER", "rounds")
CACHE_PARTITION_WINDOW = int(os.getenv("CACHE_PARTITION_WINDOW", "30"))
CACHE_TTL = os.getenv("CACHE_TTL", "5m")
PARTITION_SESSION_ID = os.getenv("PARTITION_SESSION_ID", "")


def make_cache_control() -> dict:
    if CACHE_TTL == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}

def get_active_session_id() -> str:
    return PARTITION_SESSION_ID

# 时区偏移（小时），用于记忆注入时的日期显示，默认 UTC+8
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))

_round_counter = 0

# 强制流式传输
FORCE_STREAM = os.getenv("FORCE_STREAM", "false").lower() == "true"

# 推理/思维链参数
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")

# 记忆模型专用 API Key
MEMORY_API_KEY = os.getenv("MEMORY_API_KEY", "")

def get_memory_api_key() -> str:
    return MEMORY_API_KEY or API_KEY

EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")

# 布尔解析辅助函数
def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")

# ============================================================
# 人设加载
# ============================================================

def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
_DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")

_cached_system_prompt = None
_cached_system_prompt_loaded = False

async def get_system_prompt() -> str:
    global _cached_system_prompt, _cached_system_prompt_loaded
    if _cached_system_prompt_loaded:
        return _cached_system_prompt or ""
    try:
        db_prompt = await get_gateway_config("systemPrompt", "")
        if db_prompt:
            _cached_system_prompt = db_prompt
        else:
            _cached_system_prompt = _DEFAULT_SYSTEM_PROMPT
            if _DEFAULT_SYSTEM_PROMPT:
                await set_gateway_config("systemPrompt", _DEFAULT_SYSTEM_PROMPT)
        _cached_system_prompt_loaded = True
        return _cached_system_prompt or ""
    except Exception:
        _cached_system_prompt = _DEFAULT_SYSTEM_PROMPT
        _cached_system_prompt_loaded = True
        return _cached_system_prompt or ""

def invalidate_system_prompt_cache():
    global _cached_system_prompt, _cached_system_prompt_loaded
    _cached_system_prompt = None
    _cached_system_prompt_loaded = False


# ============================================================
# 应用生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PARTITION_SESSION_ID
    if MEMORY_ENABLED:
        try:
            await init_tables()
            await ensure_token_usage_table()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
            
            try:
                db_cfg = await get_all_gateway_config()
                if db_cfg:
                    _RESTORE_MAIN = {
                        "API_BASE_URL": str, "API_KEY": str, "DEFAULT_MODEL": str,
                        "MEMORY_ENABLED": lambda v: _parse_bool(v),
                        "MAX_MEMORIES_INJECT": int, "MEMORY_EXTRACT_INTERVAL": int,
                        "CACHE_PARTITION_ENABLED": lambda v: _parse_bool(v),
                        "CACHE_PARTITION_X": int, "CACHE_PARTITION_TRIGGER": str,
                        "CACHE_PARTITION_WINDOW": int, "CACHE_SUMMARY_MODEL": str,
                        "CACHE_TTL": str,
                        "FORCE_STREAM": lambda v: _parse_bool(v),
                        "REASONING_EFFORT": str,
                    }
                    _RESTORE_DB = {
                        "EMBEDDING_API_KEY": str, "EMBEDDING_BASE_URL": str,
                        "EMBEDDING_MODEL": str, "EMBEDDING_DIM": int,
                        "MIN_SCORE_THRESHOLD": float,
                        "MEMORY_VECTOR_ENABLED": lambda v: _parse_bool(v),
                        "MEMORY_HW_KEYWORD": float, "MEMORY_HW_SEMANTIC": float,
                        "MEMORY_HW_IMPORTANCE": float, "MEMORY_HW_RECENCY": float,
                        "MEMORY_SEMANTIC_THRESHOLD": float,
                    }
                    _ALLOW_EMPTY = {"CACHE_SUMMARY_MODEL"}
                    restored = []
                    for key, val in db_cfg.items():
                        if not val:
                            if key in _ALLOW_EMPTY and key in _RESTORE_MAIN:
                                globals()[key] = _RESTORE_MAIN[key]("")
                                restored.append(key + "(显式空)")
                            continue
                        if key in _RESTORE_MAIN:
                            globals()[key] = _RESTORE_MAIN[key](val)
                            restored.append(key)
                        elif key in _RESTORE_DB:
                            setattr(_db_module, key, _RESTORE_DB[key](val))
                            restored.append(key)
                        elif key == "MEMORY_MODEL":
                            os.environ["MEMORY_MODEL"] = str(val)
                            restored.append(key)
                        elif key == "MEMORY_API_KEY":
                            globals()[key] = str(val)
                            import memory_extractor as _me_mod
                            _me_mod.MEMORY_API_KEY = str(val)
                            restored.append(key)
                    if restored:
                        print(f"🔄 从数据库恢复 {len(restored)} 项面板配置: {', '.join(restored)}")
            except Exception as e:
                print(f"[warning] 恢复面板配置失败: {e}")
            
            if not MEMORY_EXTRACT_ENABLED:
                print(f"ℹ️  记忆提取+注入已关闭（MEMORY_EXTRACT_ENABLED=false）")
            
            if CACHE_PARTITION_ENABLED:
                db_sid = await get_gateway_config("partition_session_id", "")
                if db_sid:
                    PARTITION_SESSION_ID = db_sid
                    print(f"🔗 活跃对话线(DB): {PARTITION_SESSION_ID}")
                elif PARTITION_SESSION_ID:
                    await set_gateway_config("partition_session_id", PARTITION_SESSION_ID)
                    print(f"🔗 活跃对话线(ENV→DB): {PARTITION_SESSION_ID}")
                print(f"🔒 分区缓存已启用: X={CACHE_PARTITION_X}, 摘要模型={CACHE_SUMMARY_MODEL or '（未配置，纯轮转模式）'}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
            print("⚠️  记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️  记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")
    
    yield
    
    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================
# 网关鉴权中间件
# ============================================================

PUBLIC_PATHS = ("/", "/static/", "/health", "/favicon.ico")

@app.middleware("http")
async def gateway_auth_middleware(request: Request, call_next):
    if not GATEWAY_SECRET:
        if not hasattr(gateway_auth_middleware, "_warned"):
            print("⚠️  GATEWAY_SECRET 未设置！所有 API 端点不受保护！")
            gateway_auth_middleware._warned = True
        return await call_next(request)

    path = request.url.path
    if path == "/":
        return await call_next(request)
    for prefix in PUBLIC_PATHS[1:]:
        if path.startswith(prefix):
            return await call_next(request)

    if request.method == "OPTIONS":
        return await call_next(request)

    provided_key = (
        request.headers.get("X-Gateway-Key", "")
        or request.query_params.get("gateway_key", "")
    )

    if not secrets.compare_digest(provided_key, GATEWAY_SECRET):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized. Provide X-Gateway-Key header or gateway_key parameter."},
        )

    return await call_next(request)


# ============================================================
# 记忆注入
# ============================================================

async def build_system_prompt_with_memories(user_message: str) -> str:
    if not MEMORY_ENABLED or not MEMORY_EXTRACT_ENABLED or MAX_MEMORIES_INJECT <= 0:
        return SYSTEM_PROMPT
    
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        if not memories:
            return SYSTEM_PROMPT
        
        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")
        memory_text = "\n".join(memory_lines)
        
        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."

# 交流方式
- 自然引用："记得你说过..."或"上次我们聊到..."
- 避免机械式表达如"根据我的记忆..."或"检索到的信息显示..."
- 共同经历可温情回忆："上次那个事挺好玩的"

记忆是丰富对话的工具，而非对话焦点。"""
        
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt
        
    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT


# ============================================================
# 分区缓存（Partition Cache）
# ============================================================

def _is_anthropic_model(model: str) -> bool:
    model_lower = model.lower()
    return "claude" in model_lower or "anthropic" in model_lower


def _strip_cache_control(messages: list):
    stripped = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                stripped += 1
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            msg["content"] = content[0]["text"]
    if stripped > 0:
        print(f"🔧 兼容性处理: 剥离了 {stripped} 个 cache_control 字段（非 Claude 模型）")


def _assemble_current_user_message(parts: list, raw_content) -> dict:
    if isinstance(raw_content, list):
        media_blocks = [
            b for b in raw_content
            if not (isinstance(b, dict) and b.get("type") == "text")
        ]
        text_joined = " ".join(
            b.get("text", "") for b in raw_content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if media_blocks:
            merged = "\n\n".join(parts + ([text_joined] if text_joined else []))
            return {"role": "user", "content": media_blocks + [{"type": "text", "text": merged}]}
        raw_content = text_joined
    parts.append(raw_content)
    return {"role": "user", "content": "\n\n".join(parts)}


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _is_title_generation_request(messages: list) -> bool:
    user_texts = [
        _message_text(message).strip()
        for message in messages
        if message.get("role") == "user"
    ]
    user_texts = [text for text in user_texts if text]
    if len(user_texts) != 1:
        return False

    text = user_texts[0].lower()
    strong_signatures = (
        "summarize the conversation between user and assistant into a short title",
        "summarize the conversation into a short title",
        "generate a concise title for the conversation",
        "generate a short title for the conversation",
    )
    if any(signature in text for signature in strong_signatures):
        return True

    marker_groups = (
        ("<content>", "</content>"),
        ("reply directly with the title", "only output the title", "只输出标题", "直接输出标题"),
        ("title should not exceed", "title must not exceed", "标题不超过", "标题不得超过"),
        ("conversation between user and assistant", "dialogue between user and assistant", "用户和助手的对话", "用户与助手的对话"),
        ("short title", "concise title", "简短标题", "简洁标题"),
    )
    matched_groups = sum(
        1 for markers in marker_groups if any(marker in text for marker in markers)
    )
    return matched_groups >= 3


MEMORY_USAGE_GUIDE = """

# 记忆应用
用户消息中的 <retrieved_memories> 块是网关自动检索的过往记忆，使用时：
- 像朋友般自然运用，不刻意展示；仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."
- 自然引用："记得你说过..."，避免机械式表达如"根据检索到的信息..."
"""


def build_time_injection() -> str:
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=TIMEZONE_HOURS)
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_names[now_local.weekday()]
    time_str = now_local.strftime("%Y年%m月%d日 %H:%M")
    return (
        f"<gateway_context>当前时间：{time_str} {weekday}。"
        f"此块由网关自动注入，不是用户发送的内容，无需回应或提及；"
        f"回答涉及日期、年份、时间时以此为准。</gateway_context>"
    )


async def generate_summary(messages: list, session_id: str = "") -> str:
    if not messages:
        return ""
    if not CACHE_SUMMARY_MODEL:
        print("📝 摘要模型未配置，跳过摘要生成（纯轮转模式：A区直接滑出上下文）")
        return ""
    
    conversation_text = ""
    for msg in messages:
        role_label = "用户" if msg['role'] == 'user' else "AI"
        content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
        conversation_text += f"{role_label}: {content}\n\n"
    
    prompt = f"""请将以下对话压缩成摘要。这份摘要会作为AI的记忆注入后续对话，请以AI的第一人称视角叙述（"我"指AI，用户用对话中的称呼）。
优先保留：情感节点、关系里程碑、双方的约定和决定、正在进行的话题。
保留双方的关键原话，用引号标注是谁说的。
去掉日常寒暄和重复内容。控制在300字以内。

---
{conversation_text}
---

摘要："""
    
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        if "openrouter" in API_BASE_URL:
            headers["HTTP-Referer"] = EXTRA_REFERER
            headers["X-Title"] = EXTRA_TITLE

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(API_BASE_URL, headers=headers, json={
                "model": CACHE_SUMMARY_MODEL,
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            })
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"].get("content") or ""
                    summary = content.strip()
                    if summary:
                        print(f"📝 摘要生成完成: {len(summary)}字 (压缩{len(messages)}条消息)")
                        return summary
                    print(f"⚠️ 摘要生成失败: 模型返回空content，本次轮转将推迟重试")
                    return ""

        print(f"⚠️ 摘要生成失败: HTTP {response.status_code}")
        return ""
    except Exception as e:
        print(f"⚠️ 摘要生成异常: {e}")
        return ""


def group_by_rounds(history: list) -> list:
    rounds = []
    current_round = []
    for msg in history:
        if msg['role'] == 'user' and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)
    if current_round:
        rounds.append(current_round)
    return rounds


def _should_rotate(b_rounds_count: int, X: int, a_msgs: list) -> bool:
    if b_rounds_count == 0:
        return False
    
    if CACHE_PARTITION_TRIGGER == "time":
        a_first_time = None
        for msg in a_msgs:
            t = msg.get('created_at')
            if t:
                a_first_time = t
                break
        
        if a_first_time:
            now = datetime.now(timezone.utc)
            if a_first_time.tzinfo is None:
                a_first_time = a_first_time.replace(tzinfo=timezone.utc)
            age_minutes = (now - a_first_time).total_seconds() / 60
            return age_minutes >= CACHE_PARTITION_WINDOW
        
        return b_rounds_count >= X
    
    return b_rounds_count >= X

CACHE_MAX_ROTATIONS = int(os.getenv("CACHE_MAX_ROTATIONS", "2"))


def _apply_breakpoint(msg: dict) -> bool:
    content = msg.get('content')
    if isinstance(content, str) and content.strip():
        msg['content'] = [{"type": "text", "text": content, "cache_control": make_cache_control()}]
        return True
    
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                block["cache_control"] = make_cache_control()
                return True
    return False


async def build_partitioned_messages(
    session_id: str,
    all_messages: list,
    base_prompt: str,
    user_message: str,
) -> list:
    X = CACHE_PARTITION_X
    non_system = [m for m in all_messages if m.get('role') != 'system']
    
    current_user_msg = None
    history = non_system[:]
    if history and history[-1].get('role') == 'user':
        current_user_msg = history.pop()
    
    cleaned = []
    orphan_count = 0
    for msg in history:
        if msg.get('role') == 'tool':
            prev = cleaned[-1] if cleaned else None
            if prev and (prev.get('role') == 'tool' or 
                        (prev.get('role') == 'assistant' and prev.get('tool_calls'))):
                cleaned.append(msg)
            else:
                orphan_count += 1
        else:
            cleaned.append(msg)
    if orphan_count > 0:
        print(f"⚠️ 清理了 {orphan_count} 条孤立tool消息")
    history = cleaned
    
    rounds = group_by_rounds(history)
    total_rounds = len(rounds)
    
    state = await get_session_cache_state(session_id)
    summary_parts = state['summary_parts']
    a_start_round = state['a_start_round']
    
    if total_rounds < X:
        return await _build_basic_cached(history, base_prompt, user_message, current_user_msg, summary_parts)
    
    a_end_round = a_start_round + X
    a_round_groups = rounds[a_start_round : a_end_round]
    b_round_groups = rounds[a_end_round :]
    a_msgs = [msg for rnd in a_round_groups for msg in rnd]
    b_msgs = [msg for rnd in b_round_groups for msg in rnd]
    b_rounds_count = len(b_round_groups)
    
    rotation_count = 0
    max_rotations = CACHE_MAX_ROTATIONS if CACHE_PARTITION_TRIGGER == "time" else 999
    while _should_rotate(b_rounds_count, X, a_msgs) and rotation_count < max_rotations:
        rotation_count += 1
        trigger_info = f"B区{b_rounds_count}轮 >= X={X}" if CACHE_PARTITION_TRIGGER != "time" else f"A区首条消息超出{CACHE_PARTITION_WINDOW}分钟窗口"
        print(f"🔄 轮转#{rotation_count}: session={session_id}, {trigger_info}")
        
        new_summary = await generate_summary(a_msgs, session_id)
        if new_summary:
            summary_parts.append(new_summary)
        elif CACHE_SUMMARY_MODEL:
            rotation_count -= 1
            print(f"⚠️ 摘要生成失败，本次轮转中止，下次请求重试（A区消息未丢失）")
            break

        a_start_round += X
        a_end_round = a_start_round + X
        a_round_groups = rounds[a_start_round : a_end_round]
        b_round_groups = rounds[a_end_round :]
        a_msgs = [msg for rnd in a_round_groups for msg in rnd]
        b_msgs = [msg for rnd in b_round_groups for msg in rnd]
        b_rounds_count = len(b_round_groups)
    
    if rotation_count > 0:
        await save_session_cache_state(session_id, summary_parts, a_start_round)
        summary_total = sum(len(p) for p in summary_parts)
        print(f"🔄 轮转完成(共{rotation_count}次): 摘要{len(summary_parts)}段/{summary_total}字, A区{len(a_msgs)}条, B区{len(b_msgs)}条")
    
    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": make_cache_control()}]
        })
    
    if summary_parts:
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(summary_parts):
            item = {"type": "text", "text": part}
            if i == len(summary_parts) - 1:
                item["cache_control"] = make_cache_control()
            blocks.append(item)
        result.append({"role": "user", "content": blocks})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})
    
    cleaned_a = []
    for msg in a_msgs:
        if msg.get('role') == 'tool':
            continue
        m = {k: v for k, v in msg.items() if k not in ('created_at', 'tool_calls')}
        if m.get('role') == 'assistant' and not (m.get('content') or '').strip():
            continue
        cleaned_a.append(m)
    
    for j in range(len(cleaned_a) - 1, -1, -1):
        if cleaned_a[j].get('role') != 'tool' and _apply_breakpoint(cleaned_a[j]):
            break
    
    for m in cleaned_a:
        result.append(m)
    
    b_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in b_msgs]
    for j in range(len(b_cleaned) - 1, -1, -1):
        if b_cleaned[j].get('role') != 'tool' and _apply_breakpoint(b_cleaned[j]):
            break
    
    for m in b_cleaned:
        result.append(m)
    
    if current_user_msg:
        parts = [build_time_injection()]
        if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message:
            mem_text = await build_memory_text(user_message)
            if mem_text:
                parts.append(mem_text)
        result.append(_assemble_current_user_message(parts, current_user_msg['content']))

    bp_count = 1 + (1 if summary_parts else 0) + (1 if cleaned_a else 0) + (1 if b_msgs else 0)
    summary_total = sum(len(p) for p in summary_parts)
    tool_stripped = len(a_msgs) - len(cleaned_a)
    a_info = f"A区{len(cleaned_a)}条({len(a_round_groups)}轮)" + (f"[剥离{tool_stripped}条tool]" if tool_stripped else "")
    print(f"🔒 分区缓存: BP×{bp_count} | 摘要{'有' if summary_parts else '无'}({len(summary_parts)}段/{summary_total}字) | {a_info} | B区{len(b_msgs)}条({b_rounds_count}轮) | 总{len(result)}条messages")
    return result


async def _build_basic_cached(
    history: list,
    base_prompt: str,
    user_message: str,
    current_user_msg: dict,
    summary_parts: list = None,
) -> list:
    summary_parts = summary_parts or []
    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": make_cache_control()}]
        })

    if summary_parts:
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(summary_parts):
            item = {"type": "text", "text": part}
            if i == len(summary_parts) - 1:
                item["cache_control"] = make_cache_control()
            blocks.append(item)
        result.append({"role": "user", "content": blocks})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})
    
    h_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in history]
    for j in range(len(h_cleaned) - 1, -1, -1):
        if h_cleaned[j].get('role') != 'tool' and _apply_breakpoint(h_cleaned[j]):
            break
    
    for m in h_cleaned:
        result.append(m)
    
    if current_user_msg:
        parts = [build_time_injection()]
        if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message:
            mem_text = await build_memory_text(user_message)
            if mem_text:
                parts.append(mem_text)
        result.append(_assemble_current_user_message(parts, current_user_msg['content']))

    summary_total = sum(len(p) for p in summary_parts)
    bp_count = 1 + (1 if summary_parts else 0) + (1 if history else 0)
    print(f"🔒 基础缓存(降级): BP×{bp_count} | 摘要{'有' if summary_parts else '无'}({len(summary_parts)}段/{summary_total}字) | 历史{len(history)}条 | 总{len(result)}条messages")
    return result


async def build_memory_text(user_message: str) -> str:
    if MAX_MEMORIES_INJECT <= 0:
        return ""
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        if not memories:
            return ""
        
        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")
        
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return (
            "<retrieved_memories>\n"
            "以下是网关从过往对话中自动检索的相关记忆，供参考，非用户本次输入：\n"
            + "\n".join(memory_lines)
            + "\n</retrieved_memories>"
        )
    except Exception as e:
        print(f"⚠️ 记忆检索失败: {e}")
        return ""


# ============================================================
# 后台记忆处理
# ============================================================

async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str, context_messages: list = None, skip_conversation_log: bool = False, tool_messages: list = None, assistant_tool_calls: list = None, assistant_reasoning: str = None):
    global _round_counter
    try:
        print(f"💾 process_memories_background: user_msg={bool(user_msg)}, tool_messages={len(tool_messages) if tool_messages else 0}, "
              f"assistant_tool_calls={len(assistant_tool_calls) if assistant_tool_calls else 0}, skip={skip_conversation_log}")
        
        if skip_conversation_log:
            print(f"⏭️  跳过对话存储（辅助请求）")
        elif tool_messages:
            for tm in tool_messages:
                meta_dict = {}
                if tm.get("tool_call_id"):
                    meta_dict["tool_call_id"] = tm["tool_call_id"]
                if tm.get("name"):
                    meta_dict["name"] = tm["name"]
                meta = json.dumps(meta_dict) if meta_dict else None
                await save_message(session_id, "tool", tm.get("content", ""), model, metadata=meta)
            
            if assistant_msg or assistant_tool_calls:
                ast_meta_dict = {}
                if assistant_tool_calls:
                    ast_meta_dict["tool_calls"] = assistant_tool_calls
                if assistant_reasoning:
                    ast_meta_dict["reasoning_content"] = assistant_reasoning
                ast_meta = json.dumps(ast_meta_dict) if ast_meta_dict else None
                await save_message(session_id, "assistant", assistant_msg or "", model, metadata=ast_meta)
                print(f"🔧 存储: {len(tool_messages)}条tool + 1条assistant")
        else:
            ast_meta_dict = {}
            if assistant_tool_calls:
                ast_meta_dict["tool_calls"] = assistant_tool_calls
            if assistant_reasoning:
                ast_meta_dict["reasoning_content"] = assistant_reasoning
            assistant_meta = json.dumps(ast_meta_dict) if ast_meta_dict else None
            
            if assistant_tool_calls:
                await save_message(session_id, "user", user_msg, model)
                await save_message(session_id, "assistant", assistant_msg or "", model, metadata=assistant_meta)
            else:
                last_user = await get_last_user_content(session_id)
                if last_user and last_user.strip() == user_msg.strip():
                    updated = await update_last_assistant_message(session_id, assistant_msg, model)
                    if updated:
                        print(f"🔄 检测到re-roll，已覆盖最后一条assistant回复")
                    else:
                        await save_message(session_id, "user", user_msg, model)
                        await save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)
                else:
                    await save_message(session_id, "user", user_msg, model)
                    await save_message(session_id, "assistant", assistant_msg, model, metadata=assistant_meta)
        
        if not MEMORY_EXTRACT_ENABLED or MEMORY_EXTRACT_INTERVAL == 0:
            return
        
        _round_counter += 1
        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            return
        
        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]
        
        if context_messages:
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [{"role": "assistant", "content": assistant_msg}]
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        
        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)
        
        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署", "bug", "debug", "端口", "网关",
        ]
        
        filtered_memories = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                continue
            filtered_memories.append(mem)
        
        for mem in filtered_memories:
            await save_memory(content=mem["content"], importance=mem["importance"], source_session=session_id)
        
        if filtered_memories:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered_memories)} 条新记忆，总计 {total} 条")
            
    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")


# ============================================================
# API 接口
# ============================================================

@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try: memory_count = await get_all_memories_count()
        except: pass
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v2.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ai-memory-gateway",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY 未设置"})
    try:
        return await _chat_completions_inner(request)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "gateway_error"}})


async def _chat_completions_inner(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    explicit_skip = request.headers.get("X-Skip-Conversation-Log", "").lower() == "true"
    auxiliary_title_request = _is_title_generation_request(messages)
    skip_conversation_log = explicit_skip or auxiliary_title_request
    
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            break
    
    original_messages = [msg for msg in messages if msg.get("role") != "system"]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    
    session_id = str(uuid.uuid4())[:8]
    
    if CACHE_PARTITION_ENABLED and not skip_conversation_log:
        active_sid = get_active_session_id()
        if active_sid:
            session_id = active_sid
        
        try:
            db_history = await get_conversation_messages(session_id, limit=10000)
            db_msgs = []
            for m in (db_history or []):
                msg = db_row_to_message(m)
                msg['created_at'] = m.get('created_at')
                db_msgs.append(msg)
        except Exception as e:
            print(f"[warning] 读取历史失败: {e}")
            db_msgs = []
        
        client_new_msgs = [m for m in messages if m.get("role") != "system" and m.get("role") != "assistant"]
        
        tail_user_ids = set()
        for m in reversed([m for m in messages if m.get("role") != "system"]):
            if m.get("role") == "user": tail_user_ids.add(id(m))
            else: break
        
        user_msgs = [m for m in client_new_msgs if m.get("role") == "user"]
        if len(user_msgs) > len(tail_user_ids):
            client_new_msgs = [m for m in client_new_msgs if m.get("role") != "user" or id(m) in tail_user_ids]
            
        client_tools = [m for m in client_new_msgs if m.get("role") == "tool"]
        if client_tools:
            db_last = db_msgs[-1] if db_msgs else None
            db_expecting_tool = (db_last and db_last.get("role") == "assistant" and db_last.get("tool_calls"))
            
            if not db_expecting_tool:
                client_new_msgs = [m for m in client_new_msgs if m.get("role") != "tool" ]
            else:
                expected_tool_ids = {tc.get("id") for tc in db_last.get("tool_calls", []) if tc.get("id")}
                new_tools = [m for m in client_tools if m.get("tool_call_id") in expected_tool_ids]
                tail_users = [m for m in client_new_msgs if m.get("role") == "user"]
                client_new_msgs = new_tools[:] + tail_users
                
                if new_tools:
                    new_tool_ids = {m.get("tool_call_id") for m in new_tools if m.get("tool_call_id")}
                    db_has_matching_ast = any(m.get("role") == "assistant" and m.get("tool_calls") and (new_tool_ids & {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}) for m in db_msgs)
                    if not db_has_matching_ast:
                        for m in messages:
                            if m.get("role") == "assistant" and m.get("tool_calls") and (new_tool_ids & {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}):
                                client_new_msgs.insert(0, m)
                                break
        
        all_msgs = db_msgs + client_new_msgs
        tool_messages = [m for m in client_new_msgs if m.get("role") == "tool"]
        
        partition_prompt = SYSTEM_PROMPT
        if MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and MAX_MEMORIES_INJECT > 0:
            partition_prompt = (SYSTEM_PROMPT or "") + MEMORY_USAGE_GUIDE
            
        messages = await build_partitioned_messages(session_id, all_msgs, partition_prompt, user_message)
        body["messages"] = messages
    else:
        if not skip_conversation_log and (SYSTEM_PROMPT or (MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message)):
            enhanced_prompt = await build_system_prompt_with_memories(user_message) if (MEMORY_ENABLED and MEMORY_EXTRACT_ENABLED and user_message) else SYSTEM_PROMPT
            if enhanced_prompt:
                has_system = any(msg.get("role") == "system" for msg in messages)
                if has_system:
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                            break
                else:
                    messages.insert(0, {"role": "system", "content": enhanced_prompt})
        body["messages"] = messages
    
    model = body.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
    body["model"] = model
    
    if CACHE_PARTITION_ENABLED and not _is_anthropic_model(model):
        _strip_cache_control(body.get("messages", []))
        
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE
        
    is_stream = body.get("stream", False) or FORCE_STREAM
    if FORCE_STREAM: body["stream"] = True
    
    if REASONING_EFFORT and not skip_conversation_log:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = REASONING_EFFORT
        
    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages, skip_conversation_log, tool_messages),
            media_type="text/event-stream",
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg, assistant_tool_calls, assistant_reasoning = "", None, None
                try:
                    msg_obj = resp_data["choices"][0]["message"]
                    assistant_msg = msg_obj.get("content") or ""
                    assistant_tool_calls = msg_obj.get("tool_calls")
                    assistant_reasoning = msg_obj.get("reasoning_content")
                except: pass
                
                if MEMORY_ENABLED and (user_message or tool_messages):
                    asyncio.create_task(process_memories_background(session_id, user_message, assistant_msg, model, original_messages, skip_conversation_log, tool_messages, assistant_tool_calls, assistant_reasoning))
                return JSONResponse(status_code=200, content=resp_data)
            return JSONResponse(status_code=response.status_code, content=response.json() if response.headers.get("content-type","").startswith("application/json") else {"error": response.text})


async def stream_and_capture(headers: dict, body: dict, session_id: str, user_message: str, model: str, original_messages: list = None, skip_conversation_log: bool = False, tool_messages: list = None):
    full_response, full_reasoning, stream_usage = [], [], {}
    line_buffer = ""
    accumulated_tool_calls = {}
    
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            is_error = response.status_code != 200
            async for chunk in response.aiter_bytes():
                yield chunk
                if is_error: continue
                
                text = chunk.decode("utf-8", errors="ignore")
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            if "usage" in data: stream_usage = data["usage"]
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            
                            content = delta.get("content", "")
                            if content: full_response.append(content)
                            
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning: full_reasoning.append(reasoning)
                            
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {"index": idx, "id": tc.get("id", ""), "type": "function", "function": {"name": "", "arguments": ""}}
                                    if tc.get("id"): accumulated_tool_calls[idx]["id"] = tc["id"]
                                    if "function" in tc:
                                        if tc["function"].get("name"): accumulated_tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                                        if tc["function"].get("arguments"): accumulated_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                        except: pass
                        
    assistant_msg = "".join(full_response)
    assistant_reasoning = "".join(full_reasoning) if full_reasoning else None
    assistant_tool_calls = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None
    
    if stream_usage and stream_usage.get("total_tokens", 0) > 0 and not skip_conversation_log:
        asyncio.create_task(save_token_usage(session_id, model, stream_usage.get("prompt_tokens", 0), stream_usage.get("completion_tokens", 0), stream_usage.get("total_tokens", 0)))
        
    if MEMORY_ENABLED and (user_message or tool_messages):
        asyncio.create_task(process_memories_background(session_id, user_message, assistant_msg, model, original_messages, skip_conversation_log, tool_messages, assistant_tool_calls, assistant_reasoning))


# ============================================================
# 记忆管理接口、三层记忆架构整理 API 
# ============================================================

@app.get("/import/seed-memories")
async def import_seed_memories():
    try:
        from seed_memories import run_seed_import
        return await run_seed_import()
    except ImportError: return {"error": "未找到 seed_memories.py"}
    except Exception as e: return {"error": str(e)}

@app.get("/export/memories")
async def export_memories():
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    try:
        memories = await get_all_memories()
        for mem in memories:
            if mem.get("created_at"): mem["created_at"] = str(mem["created_at"])
        return {"total": len(memories), "exported_at": str(datetime.now()), "memories": memories}
    except Exception as e: return {"error": str(e)}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not MEMORY_ENABLED: return HTMLResponse("<h3>记忆系统未启用</h3>")
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/api/memories")
async def api_get_memories(layer: int = None, active_only: bool = None):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    memories = await get_all_memories_detail(layer=layer, active_only=active_only)
    tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    try: layer_stats = await get_layer_statistics()
    except: layer_stats = None
    return {"memories": memories, "layer_stats": layer_stats}

@app.get("/api/memories/search")
async def api_search_memories(q: str = "", limit: int = 20):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    if not q.strip(): return {"error": "关键词不能为空", "results": []}
    try:
        results = await search_memories(q.strip(), limit)
        tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
        out = []
        for r in results:
            item = dict(r)
            if item.get("created_at"):
                dt = item["created_at"]
                if hasattr(dt, 'tzinfo'):
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    item["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
            out.append(item)
        return {"results": out, "total": len(out)}
    except Exception as e: return {"error": str(e), "results": []}

@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory_with_layer(memory_id, content=data.get("content"), importance=data.get("importance"), title=data.get("title"), layer=data.get("layer"))
    return {"status": "ok", "id": memory_id}

@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int, soft: bool = False):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    if soft: await update_memory_with_layer(memory_id, is_active=False)
    else: await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}

@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    for item in updates:
        await update_memory_with_layer(item["id"], content=item.get("content"), importance=item.get("importance"), title=item.get("title"), layer=item.get("layer"))
    return {"status": "ok", "updated": len(updates)}

@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids: return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


CONSOLIDATION_PROMPT = """
你是记忆整理助手。请将以下对话碎片整理成完整的事件记录。
...（保留你的原始PROMPT规范，略去不占篇幅）...
"""

_consolidate_status = {"running": False, "started_at": None, "result": None, "error": None}

async def consolidate_memories_for_date_range(start_date, end_date):
    import re
    fragments = await get_fragments_by_date_range(start_date, end_date)
    if not fragments: return {"status": "no_fragments", "start_date": str(start_date), "end_date": str(end_date)}
    
    fragments_text = "\n".join([f"[ID={f['id']}] ({f['created_at'].strftime('%m-%d') if hasattr(f['created_at'], 'strftime') else str(f['created_at'])[:10]}) {f['content']}" for f in fragments])
    prompt = CONSOLIDATION_PROMPT.format(fragments=fragments_text)
    consolidation_model = os.getenv("MEMORY_MODEL", "") or os.getenv("DEFAULT_MODEL", "anthropic/claude-haiku-4.5")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(API_BASE_URL, headers={"Authorization": f"Bearer {get_memory_api_key()}", "Content-Type": "application/json"}, json={"model": consolidation_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000})
            if response.status_code != 200: return {"status": "error", "error": f"HTTP {response.status_code}"}
            
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            json_match = re.search(r'\[[\s\S]*\]', content)
            if not json_match: return {"status": "error", "error": "AI未返回有效JSON"}
            
            events = json.loads(json_match.group())
            created_count = 0
            for event in events:
                merged_ids = event.get("merged_ids", [])
                if merged_ids:
                    await create_event_memory(title=event.get("title", ""), content=event.get("content", ""), importance=event.get("importance", 5), event_date=start_date, merged_from=merged_ids)
                    created_count += 1
            await deactivate_memories([f['id'] for f in fragments])
            return {"status": "ok", "fragments_processed": len(fragments), "events_created": created_count}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/api/memories/consolidate")
async def api_manual_consolidate(request: Request):
    if not MEMORY_ENABLED: return {"error": "记忆系统未启用"}
    if _consolidate_status.get("running"): return {"status": "already_running"}
    
    data = await request.json()
    try:
        start_date = datetime.strptime(data.get("start_date"), "%Y-%m-%d").date()
        end_date = datetime.strptime(data.get("end_date"), "%Y-%m-%d").date()
    except: return {"error": "日期格式不正确"}
    
    async def _run():
        _consolidate_status.update({"running": True, "started_at": f"{start_date}~{end_date}", "result": None, "error": None})
        try: _consolidate_status["result"] = await consolidate_memories_for_date_range(start_date, end_date)
        except Exception as e: _consolidate_status["error"] = str(e)
        finally: _consolidate_status["running"] = False
    asyncio.create_task(_run())
    return {"status": "started"}

@app.get("/api/memories/consolidate/status")
async def api_consolidate_status(): return _consolidate_status

@app.post("/api/memories/{memory_id}/promote")
async def api_promote_to_core(memory_id: int, request: Request):
    data = await request.json()
    await promote_to_core(memory_id, title=data.get("title"))
    return {"status": "ok"}

@app.post("/api/memories/merge")
async def api_merge_memories(request: Request):
    data = await request.json()
    new_id = await merge_memories(data.get("ids", []), data.get("title", ""), data.get("content", ""), data.get("importance", 5), data.get("layer", 2))
    return {"status": "ok", "new_id": new_id}

@app.post("/api/memories/check-duplicate")
async def api_check_duplicate(request: Request):
    data = await request.json()
    return await check_duplicate_memory(data.get("content", ""), data.get("threshold", 0.7))

@app.post("/api/memories/cleanup-fragments")
async def api_cleanup_fragments(request: Request):
    data = await request.json()
    deleted = await cleanup_old_fragments(data.get("days", 30))
    return {"status": "ok", "deleted": deleted}

@app.post("/api/memories/{memory_id}/revert-merge")
async def api_revert_merge(memory_id: int): return await revert_merge(memory_id)

@app.post("/api/memories/{memory_id}/restore")
async def api_restore_memory(memory_id: int):
    await update_memory_with_layer(memory_id, is_active=True)
    return {"status": "ok"}

@app.get("/api/memories/layer-stats")
async def api_layer_statistics(): return await get_layer_statistics()


# ============================================================
# 数据导入导出与历史对话管理线等 API 
# ============================================================

@app.post("/import/text")
async def import_text_memories(request: Request):
    data = await request.json()
    lines, skip_scoring = data.get("lines", []), data.get("skip_scoring", False)
    scored = [{"content": t, "importance": 5} for t in lines] if skip_scoring else await score_memories(lines)
    imported = 0
    for mem in scored:
        if not mem.get("content"): continue
        await save_memory(content=mem["content"], importance=mem.get("importance", 5), source_session="text-import")
        imported += 1
    return {"status": "done", "imported": imported}

@app.post("/import/memories")
async def import_memories(request: Request):
    data = await request.json()
    imported = 0
    for mem in data.get("memories", []):
        if not mem.get("content"): continue
        await save_memory(content=mem["content"], importance=mem.get("importance", 5), source_session=mem.get("source_session", "json-import"))
        imported += 1
    return {"status": "done", "imported": imported}

@app.get("/api/conversations")
async def api_conversations(page: int = 1, per_page: int = 20):
    results, total = await get_conversations_paginated(page, per_page)
    return {"conversations": results, "total": total, "page": page, "per_page": per_page, "total_pages": max(1, -(-total // per_page))}

@app.get("/api/conversations/{session_id}/messages")
async def api_conversation_messages(session_id: str, limit: int = 50, offset: int = 0):
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE session_id = $1", session_id)
        rows = await conn.fetch("SELECT id, role, content, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3", session_id, limit, offset)
    msgs = [{"id": r["id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"].isoformat() if r.get("created_at") else None} for r in rows]
    return {"messages": msgs, "total": total}

@app.delete("/api/conversations/{session_id}")
async def api_delete_conversation(session_id: str):
    await delete_conversation(session_id)
    return {"status": "ok"}

@app.post("/api/conversations/batch-delete")
async def api_batch_delete(request: Request):
    body = await request.json()
    await batch_delete_conversations(body.get("session_ids", []))
    return {"status": "ok"}

@app.post("/api/admin/merge-sessions")
async def api_merge_sessions(request: Request):
    body = await request.json()
    return await merge_sessions_to_target([s for s in body.get("source_ids", []) if s != body.get("target_id", "")], body.get("target_id", ""))

@app.get("/api/chat/search")
async def api_search_conversations(q: str = "", limit: int = 20, offset: int = 0):
    results, total = await search_conversations(q.strip(), limit, offset)
    return {"results": results, "total": total}

@app.patch("/api/chat/messages/{message_id}")
async def api_update_message(message_id: int, request: Request):
    body = await request.json()
    await update_message_content(message_id, body.get("content", "").strip())
    return {"status": "ok"}

@app.delete("/api/chat/messages/{message_id}")
async def api_delete_message(message_id: int):
    await delete_single_message(message_id)
    return {"status": "ok"}

@app.get("/api/conversations/export")
async def api_export_conversations():
    return JSONResponse(content=await export_all_conversations())

@app.post("/api/conversations/import")
async def api_import_conversations(request: Request):
    imported, skipped = await import_conversations(await request.json())
    return {"status": "ok", "imported": imported, "skipped": skipped}


# ============================================================
# 对话线与向量补算管理 API
# ============================================================

@app.get("/api/partition/status")
async def api_partition_status():
    active_sid = get_active_session_id()
    state = await get_session_cache_state(active_sid) if active_sid else {}
    return {
        "enabled": CACHE_PARTITION_ENABLED, "active_session_id": active_sid, "partition_x": CACHE_PARTITION_X, "summary_model": CACHE_SUMMARY_MODEL,
        "summary_parts": state.get('summary_parts', []), "summary_length": sum(len(p) for p in state.get('summary_parts', [])), "a_start_round": state.get('a_start_round', 0)
    }

@app.get("/api/partition/threads")
async def api_partition_threads():
    threads = await list_all_session_cache_states()
    active_sid = get_active_session_id()
    for t in threads: t['is_active'] = (t['session_id'] == active_sid)
    return {"threads": threads, "active_session_id": active_sid}

@app.put("/api/partition/summary")
async def api_update_summary(request: Request):
    body = await request.json()
    sid, summary = body.get("session_id"), body.get("summary", "")
    parts = [summary] if isinstance(summary, str) and summary else summary if isinstance(summary, list) else []
    await save_session_cache_state(sid, parts, 0 if not parts else 0)
    return {"status": "ok"}

@app.delete("/api/partition/summary")
async def api_clear_summary(request: Request):
    body = await request.json()
    await save_session_cache_state(body.get("session_id"), [], 0)
    return {"status": "ok"}

@app.post("/api/partition/thread")
async def api_create_thread(request: Request):
    body = await request.json()
    new_id, copy_from = body.get("session_id", "").strip(), body.get("copy_summary_from", "")
    parts = (await get_session_cache_state(copy_from)).get('summary_parts', []) if copy_from else []
    await save_session_cache_state(new_id, parts, 0)
    return {"status": "ok"}

@app.post("/api/partition/switch")
async def api_switch_thread(request: Request):
    global PARTITION_SESSION_ID
    body = await request.json()
    PARTITION_SESSION_ID = body.get("session_id", "").strip()
    await set_gateway_config("partition_session_id", PARTITION_SESSION_ID)
    return {"status": "ok"}

@app.put("/api/partition/thread/rename")
async def api_rename_thread(request: Request):
    global PARTITION_SESSION_ID
    body = await request.json()
    old_id, new_id = body.get("old_id", "").strip(), body.get("new_id", "").strip()
    if await rename_session_id(old_id, new_id):
        if PARTITION_SESSION_ID == old_id:
            PARTITION_SESSION_ID = new_id
            await set_gateway_config("partition_session_id", new_id)
    return {"status": "ok"}

@app.delete("/api/partition/thread/{session_id:path}")
async def api_delete_thread(session_id: str):
    if session_id == get_active_session_id(): return {"error": "不能删除活跃线"}
    await delete_session_cache_state(session_id)
    return {"status": "ok"}

_backfill_mem_status = {"running": False, "total": 0, "done": 0, "error": None, "finished_at": None}

@app.post("/api/admin/backfill-memory-embeddings")
async def api_backfill_memory_embeddings():
    if _backfill_mem_status["running"]: return {"error": "任务运行中"}
    total = await get_pending_memory_embedding_count()
    if total == 0: return {"status": "done", "total": 0}
    
    _backfill_mem_status.update({"running": True, "total": total, "done": 0, "error": None, "finished_at": None})
    async def run_backfill():
        try:
            while _backfill_mem_status["running"]:
                updated = await backfill_memory_embeddings(batch_size=20)
                _backfill_mem_status["done"] += updated
                if updated == 0: break
                await asyncio.sleep(1)
            _backfill_mem_status["finished_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e: _backfill_mem_status["error"] = str(e)
        finally: _backfill_mem_status["running"] = False
    asyncio.create_task(run_backfill())
    return {"status": "started", "total": total}

@app.get("/api/admin/backfill-memory-embeddings/status")
async def api_backfill_memory_embeddings_status(): return _backfill_mem_status


# ============================================================
# 模型列表 API（/api/models）—— 补算与修复完成版
# ============================================================

@app.get("/api/models")
async def get_models():
    """获取可用模型列表（根据 API_BASE_URL 自动获取或返回 Mock 兜底数据）"""
    is_openrouter = "openrouter.ai" in API_BASE_URL
    is_google = "googleapis.com" in API_BASE_URL or "generativelanguage" in API_BASE_URL
    is_openai = "api.openai.com" in API_BASE_URL

    try:
        if is_openrouter:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    simplified = [{"id": m.get("id"), "name": m.get("name"), "context_length": m.get("context_length")} for m in models]
                    simplified.sort(key=lambda x: x.get("name", ""))
                    return {"models": simplified, "total": len(simplified), "provider": "openrouter"}

        elif is_google:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    simplified = []
                    for m in models:
                        full_name = m.get("name", "")
                        model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                        if "generateContent" in m.get("supportedGenerationMethods", []):
                            simplified.append({"id": model_id, "name": m.get("displayName", model_id), "context_length": m.get("inputTokenLimit")})
                    return {"models": simplified, "total": len(simplified), "provider": "google"}

        elif is_openai:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    simplified = [{"id": m.get("id"), "name": m.get("id"), "context_length": 128000} for m in models]
                    return {"models": simplified, "total": len(simplified), "provider": "openai"}

        # 针对本地 Ollama 或自定义兼容中转的通用兜底返回
        return {
            "models": [{"id": DEFAULT_MODEL, "name": DEFAULT_MODEL, "context_length": 128000}],
            "total": 1,
            "provider": "custom_or_local"
        }
    except Exception as e:
        return {
            "error": f"无法动态拉取模型列表: {str(e)}",
            "models": [{"id": DEFAULT_MODEL, "name": DEFAULT_MODEL, "context_length": 128000}],
            "total": 1,
            "provider": "fallback"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
