import os
import time
from typing import List, Literal, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
# ========== 基础配置 ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

app = FastAPI(title="AI Relationship Coach API", version="1.0.0")

# CORS：前端本地/线上都能调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 每个IP每分钟最多 N 次
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
_call_window: dict[str, List[float]] = {}

@app.middleware("http")
async def ratelimit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _call_window.setdefault(ip, [])
    # 滤掉1分钟前的请求
    one_min_ago = now - 60
    while bucket and bucket[0] < one_min_ago:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        return HTTPException(status_code=429, detail="Too many requests, slow down.")
    bucket.append(now)
    response = await call_next(request)
    return response

# ========== 安全/边界 ==========
SAFE_SYSTEM_PROMPT = (
    "You are 'HeartMate', a supportive, practical, evidence-aware dating & relationship coach. "
    "Principles: respect boundaries, consent, kindness, honesty, mental wellbeing. "
    "Give specific, actionable suggestions (step-by-step wording, examples). "
    "Avoid explicit sexual content; do not provide illegal, manipulative, or harmful advice. "
    "When asked for therapy or diagnosis, provide gentle guidance and suggest seeking professionals. "
    "If the user is under 18 or asks about minors, refuse and suggest safe resources. "
    "Use the user's language if specified. Keep answers concise but concrete."
)

def _lang_hint(lang: Optional[str]) -> str:
    m = {
        "zh": "请用中文回答，语气真诚、尊重、实际可操作。",
        "en": "Please answer in English with a warm, respectful, practical tone.",
        "ja": "日本語で、丁寧かつ実用的に答えてください。",
        "ko": "한국어로 정중하고 실용적으로 답해 주세요。",
        "es": "Responde en español con tono respetuoso y práctico."
    }
    return m.get((lang or "zh").lower(), m["zh"])

def chat_complete(messages: List[dict], temperature: float = 0.4, max_tokens: int = 600) -> str:
    """统一的 LLM 调用封装（Groq / OpenAI 兼容）"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI backend error: {e}")

# ========== 请求/响应模型 ==========
class AdviceIn(BaseModel):
    question: str = Field(..., min_length=2, description="用户的提问或场景描述")
    context: Optional[str] = Field(None, description="可选：对象信息、过往聊天等")
    goal: Optional[str] = Field(None, description="可选：用户目标，如“邀约周末咖啡”")
    lang: Optional[str] = Field("zh", description="语言：zh/en/ja/ko/es")

class SuggestReplyIn(BaseModel):
    messages: List[dict] = Field(..., description="聊天记录，[{role:'user/other/me', content:'...'}]")
    persona: Optional[str] = Field(None, description="你的风格，例如“真诚内向理工男”")
    intent: Optional[str] = Field(None, description="目标，如：缓和气氛/转移话题/邀约")
    lang: Optional[str] = "zh"

class OpenersIn(BaseModel):
    profile_me: Optional[str] = Field(None, description="你的爱好/职业/地点等")
    profile_target: Optional[str] = Field(None, description="对方主页关键词、兴趣点")
    scene: Literal["dating_app","wechat_first","reconnect","apology","birthday","holiday"] = "dating_app"
    n: int = Field(5, ge=1, le=12)
    lang: Optional[str] = "zh"

class RewriteIn(BaseModel):
    text: str = Field(..., min_length=1)
    tone: Literal["warm","funny","confident","polite","apology","flirty_soft"] = "warm"
    keep_length: bool = False
    lang: Optional[str] = "zh"

class SentimentIn(BaseModel):
    text: str = Field(..., min_length=1)
    lang: Optional[str] = "zh"

class EQItem(BaseModel):
    id: str
    q: str           # 问题
    weight: float = 1.0
    options: List[str] = ["非常不同意","不同意","一般","同意","非常同意"]

class EQQuizReq(BaseModel):
    n: int = 10
    lang: str = "zh"             # "zh"/"en"/...
    difficulty: Optional[str] = None  # "easy|medium|hard"
    theme: Optional[str] = None       # 可选：恋爱沟通、界限、冲突…

class EQGradeReq(BaseModel):
    answers: List[int]           # 每题 1~5
    weights: Optional[List[float]] = None

@app.post("/api/eq/quiz")
def gen_eq_quiz(req: EQQuizReq):
    system = "You are an expert in relationship psychology and emotional intelligence assessment."
    user = f"""
Generate {req.n} concise EQ assessment items in {req.lang}.
Each item tests practical dating/relationship communication (active listening, boundaries, conflict de-escalation, self-regulation).
Return STRICT JSON with key "items": an array of objects:
  - id: short id (e.g., q1, q2)
  - q: question text (<= 60 chars)
  - weight: 0.5~1.5
No commentary. JSON only.
Option labels are fixed: ["非常不同意","不同意","一般","同意","非常同意"].
Difficulty: {req.difficulty or "medium"}; Theme: {req.theme or "general"}.
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.4,
            response_format={"type":"json_object"},
        )
        data = resp.choices[0].message.content
        import json
        parsed = json.loads(data)
        items = [EQItem(**it) for it in parsed.get("items", [])][:req.n]
        if not items:
            raise ValueError("empty")
        return {"items":[it.model_dump() for it in items]}
    except Exception:
        # 回退：本地题库
        fallback = [
            {"id":"q1","q":"当对方表达情绪时我会先复述确认","weight":1.0},
            {"id":"q2","q":"争执时我能控制语气避免指责","weight":1.0},
            {"id":"q3","q":"我能清晰表达需求与界限","weight":1.0},
            {"id":"q4","q":"我会观察对方的非语言线索","weight":1.0},
            {"id":"q5","q":"被拒绝时我能尊重且不施压","weight":1.0},
        ]
        return {"items":[EQItem(**it).model_dump() for it in fallback]}

@app.post("/api/eq/grade")
def grade_eq(req: EQGradeReq):
    # 简单线性计分（1~5 * weight）→ 百分制
    if not req.answers:
        raise HTTPException(400, "answers required")
    weights = req.weights or [1.0]*len(req.answers)
    if len(weights) != len(req.answers):
        raise HTTPException(400, "weights length mismatch")
    max_score = sum(w*5 for w in weights)
    got = sum(a*w for a,w in zip(req.answers, weights))
    pct = round(got/max_score*100)
    tier, advice = (
        ("A（优秀）","保持复述确认/清晰表达/尊重界限，可尝试更深层价值观对话。") if pct>=85 else
        ("B（良好）","分歧时用“我感受…我需要…我们可以…”，触发点先暂停再谈。") if pct>=70 else
        ("C（可提升）","练三点：①复述确认②非暴力沟通结构③先情绪后问题。") if pct>=55 else
        ("D（需加强）","从基础做起：记录触发点、每日一次正向反馈、拒绝给替代方案。")
    )
    return {"score": pct, "tier": tier, "advice": advice}

# ========== 健康检查 ==========
@app.get("/health")
def health():
    return {"ok": True, "model": GROQ_MODEL}

# ========== 1) 核心：恋爱建议 ==========
@app.post("/api/coach/advice")
def coach_advice(inp: AdviceIn):
    lang_note = _lang_hint(inp.lang)
    sys = SAFE_SYSTEM_PROMPT + "\n" + lang_note
    user_block = f"""User question: {inp.question}"""
    if inp.context:
        user_block += f"\nContext: {inp.context}"
    if inp.goal:
        user_block += f"\nGoal: {inp.goal}"
    guide = (
        "Output format:\n"
        "1) Quick take (1-2 lines)\n"
        "2) What to say (ready-to-send message, 1-2 options)\n"
        "3) Next steps (bulleted, concrete)\n"
        "Avoid clichés. Be specific. Keep it kind."
    )

    content = chat_complete(
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_block + "\n\n" + guide},
        ],
        temperature=0.4,
        max_tokens=700,
    )
    return {"answer": content}

# ========== 2) 回复建议（给定聊天记录） ==========
@app.post("/api/coach/suggest-reply")
def coach_suggest_reply(inp: SuggestReplyIn):
    lang_note = _lang_hint(inp.lang)
    sys = SAFE_SYSTEM_PROMPT + "\n" + lang_note
    history_lines = []
    for m in inp.messages[-20:]:
        role = m.get("role", "other")
        txt = m.get("content", "")
        history_lines.append(f"{role}: {txt}")
    history = "\n".join(history_lines)
    persona = inp.persona or "真诚、尊重界限、不过度热情，表达清晰。"
    intent = inp.intent or "自然推进关系"
    user_block = (
        f"Persona: {persona}\nIntent: {intent}\n\n"
        f"Chat history (latest last):\n{history}\n\n"
        "Task: Draft 2-3 alternative replies. Keep each under 60 Chinese characters (or ~35 English words). "
        "Give a one-line rationale after each suggestion."
    )
    content = chat_complete(
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_block}],
        temperature=0.5,
        max_tokens=600,
    )
    return {"answer": content}

# ========== 3) 开场白生成 ==========
@app.post("/api/coach/openers")
def coach_openers(inp: OpenersIn):
    lang_note = _lang_hint(inp.lang)
    sys = SAFE_SYSTEM_PROMPT + "\n" + lang_note
    user_block = (
        f"Scene: {inp.scene}\n"
        f"My profile: {inp.profile_me or '-'}\n"
        f"Target profile: {inp.profile_target or '-'}\n"
        f"Need {inp.n} openers. Constraints: respectful, specific, no pickup artistry, no cringe. "
        "Vary structures (question, observation, playful, appreciation)."
    )
    content = chat_complete(
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_block}],
        temperature=0.7,
        max_tokens=700,
    )
    return {"answer": content}

# ========== 4) 语气重写 ==========
@app.post("/api/coach/rewrite")
def coach_rewrite(inp: RewriteIn):
    lang_note = _lang_hint(inp.lang)
    sys = SAFE_SYSTEM_PROMPT + "\n" + lang_note
    keep_rule = "Keep length similar." if inp.keep_length else "You may shorten slightly if clearer."
    user_block = (
        f"Rewrite the message with tone={inp.tone}. {keep_rule}\n"
        "Preserve intent, remove pressure, keep boundaries respectful.\n\n"
        f"Original:\n{inp.text}"
    )
    content = chat_complete(
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_block}],
        temperature=0.5,
        max_tokens=400,
    )
    return {"answer": content}

# ========== 5) 基础情绪/立场分析 ==========
@app.post("/api/coach/sentiment")
def coach_sentiment(inp: SentimentIn):
    lang_note = _lang_hint(inp.lang)
    sys = SAFE_SYSTEM_PROMPT + "\n" + lang_note
    user_block = (
        "Classify the text by sentiment (positive/neutral/negative), "
        "perceived interest level (high/medium/low), and risk flags (boundaries/ghosting/argument/none). "
        "Return a compact JSON with fields {sentiment, interest, risk, one_line_reason}.\n\n"
        f"Text:\n{inp.text}"
    )
    content = chat_complete(
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_block}],
        temperature=0.2,
        max_tokens=220,
    )
    return {"analysis": content}

@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "detail": exc.detail}
    )