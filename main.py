# -*- coding: utf-8 -*-
"""语音意图插件：要不要发语音，交给 AI 判断，而不是掷骰子。

工作方式：
1. 插件加载后保持 AstrBot 内置语音开启，只屏蔽概率触发，避免随机发语音；
2. 每条会走 LLM 的消息，在发起请求的同时并发跑一次意图判断（模型可配置，留空跟随会话模型）；
3. 发送前如果判断结果是「想听语音」，就把文本回复转成语音；否则原样发文字。

判断与主回复并发，正常不增加延迟；判断失败一律按「不需要语音」处理。
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import time
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

try:
    from astrbot.api.provider import ProviderRequest
except Exception:  # 兼容旧版本
    ProviderRequest = None  # type: ignore

# 把说明塞进「本轮用户消息」需要 TextPart；不同版本位置不同，逐个试
TextPart = None  # type: ignore
for _mod_name in ("astrbot.core.agent.message", "astrbot.api.provider"):
    try:
        _mod = __import__(_mod_name, fromlist=["TextPart"])
        TextPart = getattr(_mod, "TextPart", None)  # type: ignore
    except Exception:
        TextPart = None  # type: ignore
    if TextPart is not None:
        break

try:
    from astrbot.core.message.message_event_result import MessageChain
except Exception:  # 兼容旧版本
    try:
        from astrbot.api.all import MessageChain  # type: ignore
    except Exception:
        MessageChain = None  # type: ignore

try:
    from astrbot.core.star.session_llm_manager import SessionServiceManager
except Exception:  # pragma: no cover
    SessionServiceManager = None  # type: ignore

PLUGIN_NAME = "astrbot_plugin_voice_intent"

# event extras 里存并发判断任务的 key
_TASK_KEY = "_voice_intent_task"
# 已经处理过判断的标记，避免工具循环重复触发
_FLAG_KEY = "_voice_intent_done"
# 系统提示与工具改造只做一次，避免工具循环里重复追加
_PROMPT_KEY = "_voice_intent_prompt_applied"
# 模型即使无视工具 schema 仍尝试发 record，也以这次调用作为明确的语音意图
_FORCE_VOICE_KEY = "_voice_intent_force_voice"

# 是否本轮已经是语音、并且已经通过工具发出去过
_VOICE_SENT_KEY = "_voice_intent_voice_sent"
# 本轮判定结果（是否语音），决定 send_voice 工具在不在模型眼前
_VOICE_MODE_KEY = "_voice_intent_voice_mode"
# 模型用来「主动说一句语音」的工具名
_VOICE_TOOL_NAME = "send_voice"

# 内置「给用户发消息」工具，智能体模式下模型会用它尝试自己发语音
_SEND_TOOL_NAME = "send_message_to_user"

# 追加到系统提示末尾的声明。目的只有一个：让模型不要自己去发音频，
# 也不要因为发不出去就向用户宣告「我没有语音」——语音是发送阶段自动合成的。
VOICE_AUTO_PROMPT = """

<voice_reply>
本会话的语音回复由系统自动完成：你输出的文字会在发送阶段被自动合成为语音发给用户。
你不需要、也无法自己生成或发送音频文件。
- 不要调用 send_message_to_user 发送 record / 语音 / 音频组件。
- 用户要求语音时，正常用文字回答即可，系统会把它变成语音。
- 不要说「我发不了语音」「没有 TTS 接口」「缺少音色」「沙箱未启用」这类话；
  你并不掌握语音通道的状态，这类判断一律是错误的。
</voice_reply>"""

# 只放在系统提示末尾不够：系统提示前面常挂着几万字的人格 / 框架设定，
# 模型会照人格演一个「用手机打字的人」，照样回「我发不出声音」。
# 因此再往「本轮用户消息」里贴一段，位置最靠近用户诉求，权重最高。
VOICE_USER_NOTE_MARK = "<voice_reply_notice>"
VOICE_USER_NOTE = (
    "\n" + VOICE_USER_NOTE_MARK + "\n"
    "（系统提示，用户看不到，不要复述本段）\n"
    "本轮回复会由系统在发送阶段自动合成为语音，你只负责把话写出来。\n"
    "禁止说「发不出声音 / 发不了语音 / 没法发语音 / 只能打字」这类话，"
    "也不要说自己没有语音功能——语音通道不在你的判断范围内，"
    "说这些话就是错的。\n"
    "</voice_reply_notice>"
)

# 工具模式下用这句，比上面短得多：只告诉模型「调哪个工具」，不再解释原理。
VOICE_TOOL_NOTE = (
    "\n" + VOICE_USER_NOTE_MARK + "\n"
    "（系统提示，用户看不到）本轮用户要的是语音回复。\n"
    "调用 send_voice 工具，把要对用户说的话完整放进 text；"
    "不要输出正文，也不要说自己发不出声音。\n"
    "</voice_reply_notice>"
)

# 预筛：像不像在要语音。命中才值得等判断结果（Jev 约 1~2 秒），
# 不命中就照旧后台并发判断，不阻塞主回复。
_VOICE_HINT_RE = re.compile(
    r"语音|声音|声线|嗓子|说句话|说一段|说一句|念给|念一遍|念一下|读给|读一遍"
    r"|唱|听听|开口|发条音|来一段|来句|用嘴"
)

# 执行层兜底：即使模型还是写了「发不出声音」，转语音前把这句摘掉，
# 避免出现「语音里说自己发不了语音」这种自相矛盾。
_REFUSAL_VERBS = (
    r"(?:发不出|发不了|没法发|不能发|无法发|不会发|给不了|没法用|用不了"
    r"|没有(?:语音|发声|音频)(?:功能|能力|通道|接口)"
    r"|(?:语音|发声|音频)(?:功能|能力|通道|接口)?(?:没有|未开启|不可用))"
)
_VOICE_NOUNS = r"(?:语音|声音|音频|嗓子|声线)"
_REFUSAL_HINT_RE = re.compile(r"只能(?:打字|文字)|打不了语音|不能说话")
_REFUSAL_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")

DEFAULT_INTENT_PROMPT = """你是一个意图分类器，只判断一件事：用户最新一条消息，是否在要求对方用「语音 / 声音」的形式回复。

判定为 YES 的情形：
- 直接要求发语音、发条语音、语音回我、用语音说、语音说呗
- 想听声音、想听听你的声音、说句话给我听、念给我听、唱一段、来一段
- 明确指定语音形式：朗读、配音、读出来、唱出来

判定为 NO 的情形：
- 只是顺带提到「语音 / 声音 / 音频」，但并不是要求对方发语音：讨论语音识别、音频文件、音量大小、转述别人的语音、描述一段录音
- 明确要求打字、用文字回复，或者说「别发语音」
- 与语音无关的普通聊天、提问、指令、追问

规则：
- 只输出一个词：YES 或 NO。
- 不要解释，不要标点，不要换行。
- 拿不准时输出 NO。"""


def _is_llm_result(result) -> bool:
    """只处理 LLM 产生的非流式结果；流式输出由 pipeline 自己处理。"""
    fn = getattr(result, "is_llm_result", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _parse_one(text: str) -> bool | None:
    """解析单段文本，识别不出来时返回 None。"""
    t = str(text or "").strip().upper()
    if not t:
        return None
    t = t.strip("。.!！,，:：;；\"'`*【】[]()（） \t\r\n")
    if not t:
        return None

    # 先判否定，避免「不要」被「要」抢先命中
    for no in ("NO", "NOT", "FALSE", "否", "不是", "不需要", "不要", "无", "0"):
        if t.startswith(no):
            return False
    for yes in ("YES", "TRUE", "需要", "是", "要", "有", "1", "Y"):
        if t.startswith(yes):
            return True

    # 模型不听话、写了一整句解释时的退化判断
    low = t.lower()
    has_yes = "yes" in low or "需要" in t or "应该发语音" in t
    has_no = "no" in low or "不需要" in t or "不是" in t or "不用" in t
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def _parse_decision(text: str, fallback: str = "") -> bool:
    """把模型回答解析成布尔值。

    content 为空时（reasoning 模型把 token 花在思考上）退而看思考内容的结尾，
    这是能拿到结论的最后一块地方。都识别不出来时按「不需要语音」处理。
    """
    decision = _parse_one(text)
    if decision is not None:
        return decision
    if fallback:
        decision = _parse_one(fallback[-400:])
        if decision is not None:
            return decision
    return False


class _NeverTriggerRandom:
    """只替换 ResultDecorateStage 模块里的 random，让概率 TTS 永不命中。"""

    def __init__(self, real_random):
        self._real_random = real_random

    def __getattr__(self, name):
        return getattr(self._real_random, name)

    def random(self) -> float:
        return 2.0


class _VoiceGuardSendTool:
    """send_message_to_user 的请求级代理，在执行层拦截 record。

    只修改 JSON Schema 不够：模型仍能输出 schema 之外的 record。这个代理保留
    原工具的其它消息类型，遇到 record 时不检查沙箱、不生成虚假错误，而是标记
    本轮必须语音，并让模型回到正常文字回复，由发送阶段统一做 TTS。
    """

    def __init__(self, original):
        self._original = original
        for name in (
            "name",
            "description",
            "parameters",
            "handler",
            "handler_module_path",
            "active",
            "is_background_task",
        ):
            if hasattr(original, name):
                setattr(self, name, getattr(original, name))

    async def call(self, context, **kwargs):
        messages = kwargs.get("messages")
        has_record = isinstance(messages, list) and any(
            isinstance(item, dict)
            and str(item.get("type") or "").strip().lower() == "record"
            for item in messages
        )
        if not has_record:
            return await self._original.call(context, **kwargs)

        try:
            context.context.event.set_extra(_FORCE_VOICE_KEY, True)
        except Exception:
            logger.warning("[voice_intent] 拦截语音工具后无法标记当前会话")
        logger.info("[voice_intent] 已拦截 record 工具调用，本轮最终回复将自动转为语音")
        return (
            "Voice request accepted. Do not create/upload an audio file and do not "
            "call this tool again for voice. Return your answer as normal assistant "
            "text now; the platform will automatically synthesize and send it as a "
            "voice message. Never claim that voice or the sandbox is unavailable."
        )


def _retry_delay(resp, attempt: int) -> float:
    """按指数退避计算重试等待，并遵守服务端的 Retry-After。"""
    delay = 0.4 * (2**attempt)
    try:
        value = str(resp.headers.get("retry-after") or "").strip()
    except Exception:
        return delay
    if not value:
        return delay

    try:
        server_delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            server_delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return delay

    if not math.isfinite(server_delay):
        return delay
    return max(delay, server_delay, 0.0)


@register(
    PLUGIN_NAME,
    "irislilymoon",
    "按 AI 意图判断决定发语音还是发文字：自动关掉内置概率语音，想听声音才发语音",
    "0.2.2",
)
class VoiceIntentPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 记录被本插件改过的 tts 配置对象，插件停用时还原 enable 原值
        self._patched: list[tuple[dict, bool]] = []
        # ResultDecorateStage 模块里的 random 被临时替换，停用时恢复
        self._stage_module = None
        self._stage_random_original = None

        # 热加载时 on_astrbot_loaded 不会再来一次，这里先按一次
        try:
            self._manage_builtin_tts(None)
        except Exception:
            logger.warning("[voice_intent] 初始化时接管内置语音失败，稍后重试")

    # ---------------- 生命周期 ----------------

    @filter.on_astrbot_loaded()
    async def _on_loaded(self):
        """AstrBot 完全加载后再接管一次，避免启动顺序问题。"""
        self._manage_builtin_tts(None)

    def _install_probability_guard(self) -> bool:
        """屏蔽 ResultDecorateStage 的概率 TTS，但保持 TTS 开关为开启。"""
        if self._stage_module is not None:
            return True
        try:
            from astrbot.core.pipeline.result_decorate import stage as stage_module
        except Exception:
            logger.warning("[voice_intent] 无法导入结果装饰阶段，概率 TTS 无法屏蔽")
            return False

        original = getattr(stage_module, "random", None)
        if isinstance(original, _NeverTriggerRandom):
            original = getattr(original, "_real_random", None)
        if original is None:
            logger.warning("[voice_intent] 结果装饰阶段缺少 random，概率 TTS 无法屏蔽")
            return False

        self._stage_module = stage_module
        self._stage_random_original = original
        stage_module.random = _NeverTriggerRandom(original)
        logger.info(
            "[voice_intent] 已屏蔽内置概率语音触发，TTS 状态保持开启"
        )
        return True

    def _restore_probability_guard(self) -> None:
        if self._stage_module is not None and isinstance(
            getattr(self._stage_module, "random", None), _NeverTriggerRandom
        ):
            self._stage_module.random = self._stage_random_original
        self._stage_module = None
        self._stage_random_original = None

    def _restore_builtin_tts(self) -> None:
        for tts_cfg, original in self._patched:
            try:
                tts_cfg["enable"] = original
            except Exception:
                pass
        self._patched.clear()
        self._restore_probability_guard()

    async def terminate(self):
        if self._patched or self._stage_module is not None:
            logger.info("[voice_intent] 插件停用，已恢复 AstrBot 内置语音配置")
        self._restore_builtin_tts()

    # ---------------- 接管内置语音 ----------------

    def _manage_builtin_tts(self, umo: str | None) -> None:
        """保持 TTS 开启，只屏蔽内置概率触发，由意图判断决定是否转语音。

        这样其他依赖 ``provider_tts_settings.enable`` 的功能会看到语音已开启；
        停用插件时再把 enable 恢复为本插件接管前的真实值。
        """
        if not bool(self.config.get("enabled", True)):
            self._restore_builtin_tts()
            return

        if not bool(self.config.get("disable_probability_tts", True)):
            self._restore_builtin_tts()
            return

        guard_ok = self._install_probability_guard()

        try:
            conf = self.context.get_config(umo)
        except Exception:
            return
        if not isinstance(conf, dict):
            return

        tts_cfg = conf.get("provider_tts_settings")
        if not isinstance(tts_cfg, dict):
            return

        desired_enable = bool(guard_ok)
        if any(tts_cfg is item[0] for item in self._patched):
            if bool(tts_cfg.get("enable")) != desired_enable:
                tts_cfg["enable"] = desired_enable
            return

        original = bool(tts_cfg.get("enable", False))
        self._patched.append((tts_cfg, original))
        tts_cfg["enable"] = desired_enable
        if not guard_ok:
            logger.warning(
                "[voice_intent] 概率触发屏蔽失败，临时关闭内置 TTS 以避免重复语音"
            )

    # ---------------- 语音工具（模型真正看得见的能力声明） ----------------

    @filter.llm_tool(name=_VOICE_TOOL_NAME)
    async def send_voice(self, event: AstrMessageEvent, text: str) -> str:
        """用语音把这段话讲给用户听。用户想听声音、要一条语音、让你念或唱时调用。

        Args:
            text(string): 要念给用户听的内容。用口语写，不要 Markdown、表情符号、括号动作。
        """
        content = str(text or "").strip()
        if not content:
            return "text 为空，没有发送。"
        if event.get_extra(_VOICE_SENT_KEY):
            return "本轮已经发过语音了，不要重复发送。"
        if bool(self.config.get("scrub_voice_refusal", True)):
            try:
                content = self._scrub_voice_refusal(content)
            except Exception:
                pass

        try:
            tts = await self._resolve_tts_provider(event)
        except Exception as e:
            logger.warning(f"[voice_intent] 获取 TTS 提供商失败：{e}")
            tts = None
        if tts is None:
            return "当前会话没有可用的语音提供商，请改用普通文字回复。"

        try:
            audio_path = await tts.get_audio(content)
        except Exception as e:
            logger.error(f"[voice_intent] send_voice 合成失败：{e}")
            return "语音合成失败，请改用普通文字回复。"
        if not audio_path:
            return "语音合成没有返回文件，请改用普通文字回复。"

        url = await self._register_audio_url(event, audio_path)
        if MessageChain is None:
            return "当前环境不支持发送语音，请改用普通文字回复。"

        try:
            await event.send(
                MessageChain(
                    chain=[
                        Comp.Record(
                            file=url or audio_path,
                            url=url or audio_path,
                            text=content,
                        )
                    ]
                )
            )
        except Exception as e:
            logger.error(f"[voice_intent] send_voice 发送失败：{e}")
            return "语音发送失败，请改用普通文字回复。"

        event.set_extra(_VOICE_SENT_KEY, True)
        logger.info(f"[voice_intent] send_voice 已发送语音（{len(content)} 字）")
        return (
            "语音已发送给用户。不要再重复这段内容，本轮不需要再输出正文，"
            "也不要说自己发不出声音。"
        )

    async def _register_audio_url(
        self, event: AstrMessageEvent, audio_path: str
    ) -> str | None:
        """按配置决定要不要把音频转成外链，失败就退回本地路径。"""
        try:
            conf = self.context.get_config(event.unified_msg_origin) or {}
            tts_settings = conf.get("provider_tts_settings") or {}
            callback_api_base = str(conf.get("callback_api_base") or "")
            if not bool(tts_settings.get("use_file_service", False)) or not callback_api_base:
                return None
            from astrbot.core import file_token_service

            token = file_token_service.register_file(audio_path)
            if asyncio.iscoroutine(token):
                token = await token
            return f"{callback_api_base}/api/file/{token}"
        except Exception as e:
            logger.warning(f"[voice_intent] 注册语音文件失败，改用本地路径：{e}")
            return None

    # ---------------- 意图判断（与主回复并发） ----------------

    @filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        try:
            if not bool(self.config.get("enabled", True)):
                return
            if not self._should_consider(event):
                return
            if not await self._session_tts_enabled(event):
                return

            if bool(self.config.get("use_voice_tool", True)):
                await self._sync_voice_tool(event, req)
            else:
                # 老方案：往提示词里灌说明，靠模型自觉
                self._prepare_voice_context(req)
                self._start_background_judge(event, req)
        except Exception:
            logger.error("[voice_intent] 启动意图判断失败：\n" + traceback.format_exc())

    def _start_background_judge(self, event: AstrMessageEvent, req) -> None:
        if event.get_extra(_FLAG_KEY):
            return
        if event.get_extra(_TASK_KEY) is not None:
            return
        event.set_extra(_FLAG_KEY, True)
        task = asyncio.create_task(self._judge(event, req))
        # 任务异常在这里被消费掉，避免 "exception was never retrieved" 警告
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        event.set_extra(_TASK_KEY, task)

    async def _sync_voice_tool(self, event: AstrMessageEvent, req) -> None:
        """按本轮判断结果决定 send_voice 工具是否出现在模型眼前。

        工具是模型唯一真正会读的能力清单（tools 字段），比在几万字系统提示
        末尾加一段话可靠得多。判为文字时把工具摘掉，避免模型见谁都念。
        结论缓存在事件上，工具循环的后续请求沿用，不会重复判断。
        """
        mode = event.get_extra(_VOICE_MODE_KEY)

        if mode is None:
            text = (event.message_str or "").strip()
            if _VOICE_HINT_RE.search(text):
                # 看着像要语音：值得先等判断结果，再决定工具可见性
                task = asyncio.create_task(self._judge(event, req))
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
                event.set_extra(_FLAG_KEY, True)
                event.set_extra(_TASK_KEY, task)
                mode = await self._await_judge(task)
                event.set_extra(_VOICE_MODE_KEY, bool(mode))
                logger.info(
                    f"[voice_intent] 预判为{'语音' if mode else '文字'}，"
                    f"send_voice 工具{'已提供' if mode else '已摘除'}"
                )
            else:
                # 不像要语音：不阻塞主回复，判断结果只留给发送阶段兜底
                self._start_background_judge(event, req)
                event.set_extra(_VOICE_MODE_KEY, False)
                mode = False

        # 工具循环的第二步起 req 可能是新对象，这里每次都要重新对齐
        if event.get_extra(_VOICE_SENT_KEY):
            # 已经发过了，别再让模型看见工具，免得重复发
            self._drop_voice_tool(req)
            return
        if mode:
            self._prepare_voice_context(req, inject_prompt=False)
            self._inject_user_turn_note(req, VOICE_TOOL_NOTE)
        else:
            self._drop_voice_tool(req)
            self._prepare_voice_context(req, inject_prompt=False)

    def _drop_voice_tool(self, req) -> None:
        """把 send_voice 从本次请求的工具集里摘掉（只影响这次请求）。"""
        toolset = getattr(req, "func_tool", None)
        if toolset is None:
            return
        try:
            toolset.remove_tool(_VOICE_TOOL_NAME)
        except Exception:
            logger.warning("[voice_intent] 摘除语音工具失败")

    def _prepare_voice_context(self, req, inject_prompt: bool = True) -> None:
        """让模型知道语音是自动的，并堵掉它「自己发语音」的那条路。

        模型在智能体模式下会主动调 send_message_to_user 发 record，沙箱没开就会
        收到一条错误，然后当真事告诉用户「我发不了语音」。语音其实是发送阶段
        由 TTS 合成后发出去的，模型完全看不到，所以这个误会会一直自我强化。

        ``inject_prompt=False`` 用于工具模式：不再往系统提示灌长段说明，
        只在语音轮贴一句「调用 send_voice」的短提示。
        """
        if getattr(req, _PROMPT_KEY, False):
            return
        try:
            setattr(req, _PROMPT_KEY, True)
        except Exception:
            # req 不允许挂自定义属性时仍然继续，宁可重复一次也不要漏掉
            pass

        if inject_prompt and bool(self.config.get("tell_model_voice_auto", True)):
            try:
                req.system_prompt = f"{getattr(req, 'system_prompt', '') or ''}{VOICE_AUTO_PROMPT}"
            except Exception:
                logger.warning("[voice_intent] 追加语音说明失败")
            if bool(self.config.get("voice_note_in_user_turn", True)):
                self._inject_user_turn_note(req, VOICE_USER_NOTE)

        if bool(self.config.get("strip_voice_tool", True)):
            try:
                self._strip_voice_component(req)
            except Exception:
                logger.error("[voice_intent] 屏蔽语音工具失败：\n" + traceback.format_exc())

    def _inject_user_turn_note(self, req, note: str = VOICE_USER_NOTE) -> None:
        """把语音说明贴到本轮用户消息里，绕开「人格 / 框架提示太长冲淡」的问题。"""
        if TextPart is None:
            return
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is None:
            return
        try:
            for part in parts:
                if VOICE_USER_NOTE_MARK in str(getattr(part, "text", "") or ""):
                    return
        except Exception:
            return
        try:
            parts.append(TextPart(text=note))
        except Exception:
            logger.warning("[voice_intent] 向用户消息追加语音说明失败")

    def _scrub_voice_refusal(self, text: str) -> str:
        """把「我发不出声音」这类自我否定句摘掉，其余原样保留。"""
        sentences = _REFUSAL_SENTENCE_RE.findall(text or "")
        if not sentences:
            return text
        kept: list[str] = []
        dropped = 0
        for sentence in sentences:
            hit = bool(
                re.search(_REFUSAL_VERBS, sentence)
                and re.search(_VOICE_NOUNS, sentence)
            ) or bool(_REFUSAL_HINT_RE.search(sentence))
            if hit:
                dropped += 1
                continue
            kept.append(sentence)
        if not dropped:
            return text
        cleaned = "".join(kept).strip()
        if not cleaned:
            cleaned = "嗯，这就说给你听。"
        logger.info(f"[voice_intent] 已摘掉 {dropped} 句「发不出声音」式自我否定")
        return cleaned

    def _strip_voice_component(self, req) -> None:
        """把 send_message_to_user 的 record 能力摘掉，其余类型保持原样。"""
        toolset = getattr(req, "func_tool", None)
        if toolset is None:
            return
        get_tool = getattr(toolset, "get_tool", None)
        add_tool = getattr(toolset, "add_tool", None)
        if not callable(get_tool) or not callable(add_tool):
            return

        tool = get_tool(_SEND_TOOL_NAME)
        if tool is None:
            return

        params = getattr(tool, "parameters", None)
        if not isinstance(params, dict):
            return

        # 全局工具是单例，必须拷贝后再改，否则会影响所有请求
        new_params = copy.deepcopy(params)
        try:
            items = new_params["properties"]["messages"]["items"]
            type_prop = items["properties"]["type"]
        except Exception:
            return
        if not isinstance(type_prop, dict):
            return

        type_prop["enum"] = ["plain", "image", "video", "file", "mention_user"]
        type_prop["description"] = (
            "组件类型，只能是 plain、image、video、file、mention_user。"
            "本平台不支持通过该工具发送语音（record）。"
        )
        path_prop = items["properties"].get("path")
        if isinstance(path_prop, dict):
            path_prop["description"] = (
                "File path for `image`, `video`, or `file` types. Both local path and sandbox path are supported."
            )

        clone = _VoiceGuardSendTool(tool)
        clone.parameters = new_params
        clone.description = (
            "Send message to the user. "
            "Supports `plain`, `image`, `video`, `file`, and `mention_user` components. "
            "Voice replies are produced automatically by the platform, so this tool "
            "cannot and must not be used to send `record` components."
        )
        add_tool(clone)
        if not getattr(self, "_strip_logged", False):
            self._strip_logged = True
            logger.info(
                "[voice_intent] 已为 send_message_to_user 加上 record 执行拦截，"
                "违规调用会转为自动语音回复"
            )

    def _should_consider(self, event: AstrMessageEvent) -> bool:
        if bool(self.config.get("only_private", False)):
            try:
                if not event.is_private_chat():
                    return False
            except Exception:
                pass

        text = (event.message_str or "").strip()
        if not text:
            return False
        # 指令消息不判断
        if text.startswith(("/", "!", "／", "！")):
            return False

        if not self._has_tts_provider():
            return False
        return True

    def _has_tts_provider(self) -> bool:
        try:
            return bool(self.context.get_all_tts_providers())
        except Exception:
            return True

    async def _session_tts_enabled(self, event: AstrMessageEvent) -> bool:
        """会话级 TTS 关闭时不要向模型宣称语音可用，也不要浪费判断请求。"""
        if SessionServiceManager is None:
            return True
        try:
            return bool(await SessionServiceManager.should_process_tts_request(event))
        except Exception:
            return True

    def _resolve_provider(self, umo: str):
        pid = str(self.config.get("intent_provider") or "").strip()
        if pid:
            provider = None
            try:
                provider = self.context.get_provider_by_id(pid)
            except Exception:
                provider = None

            if provider is None:
                try:
                    for cand in self.context.get_all_providers():
                        cfg = getattr(cand, "provider_config", None) or {}
                        if pid in (
                            str(cfg.get("id") or ""),
                            str(cfg.get("model") or ""),
                            str(cfg.get("provider") or ""),
                        ):
                            provider = cand
                            break
                except Exception:
                    provider = None

            if provider is not None:
                return provider
            logger.warning(
                f"[voice_intent] 找不到 Provider「{pid}」，本次回退到当前会话模型"
            )

        try:
            return self.context.get_using_provider(umo)
        except Exception:
            return None

    def _recent_dialog(self, req) -> list[str]:
        """把最近几条对话整理成「用户: xxx」这种行，供两种判断方式复用。"""
        if not bool(self.config.get("include_context", True)):
            return []

        lines: list[str] = []
        contexts = getattr(req, "contexts", None) or []
        for item in contexts[-6:]:
            try:
                role = str(item.get("role") or "")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(p.get("text") or "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                content = str(content).strip().replace("\n", " ")
            except Exception:
                continue
            if not content:
                continue
            label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(
                role, role
            )
            lines.append(f"{label}: {content[:300]}")
        return lines

    def _build_judge_prompt(self, user_text: str, req) -> str:
        parts: list[str] = []
        lines = self._recent_dialog(req)
        if lines:
            parts.append("最近对话：\n" + "\n".join(lines))
        parts.append(f"本轮用户消息：{user_text[:800]}")
        parts.append("本轮用户消息是否要求对方用语音回复？只回答 YES 或 NO。")
        return "\n\n".join(parts)

    def _build_jev_state(self, user_text: str, req) -> dict:
        """按官方建议用结构化 state，明确区分最近对话和本轮消息。"""
        state: dict = {"latest_user_message": user_text[:800]}
        lines = self._recent_dialog(req)
        if lines:
            state["recent_dialog"] = lines
        return state

    def _jev_key(self) -> str:
        return str(self.config.get("jev_api_key") or "").strip()

    async def _judge_jev(self, user_text: str, req, started: float) -> bool | None:
        """用 TypeSafe 的 Jev 做结构化判断。

        Jev 不是聊天模型：它接收一个 state 和若干「问题」，返回选项、概率和置信度。
        这里问一个 Choice 题——「用户要不要语音回复」，选项 voice / text。
        返回 None 表示这条路走不通（没 key、网络或接口出错），由调用方回退到对话模型。
        """
        api_key = self._jev_key()
        if not api_key or httpx is None:
            return None

        base = str(self.config.get("jev_base_url") or "https://api.typesafe.ai").strip()
        base = base.rstrip("/")
        if base.endswith("/systemone"):
            url = base
        elif base.endswith("/v1"):
            url = f"{base}/systemone"
        else:
            url = f"{base}/v1/systemone"

        threshold = float(self.config.get("jev_confidence", 0.6) or 0)
        timeout = float(self.config.get("judge_timeout", 8) or 8)

        payload: dict = {
            "state": self._build_jev_state(user_text, req),
            "questions": {
                "reply_mode": {
                    "type": "choice",
                    "instructions": {
                        "question": "`latest_user_message` 是否在要求对方用语音的形式回复？",
                        "focus": "只判断 `latest_user_message` 要不要听语音；`recent_dialog` 仅用于理解省略和指代，不判断情绪、语气、话题或内容好坏。",
                    },
                    "criteria": {
                        "voice": {
                            "what": "用户明确要求对方用语音/声音回应：发语音、发条语音、想听声音、说句话听听、念给我听、唱一段、用语音回答",
                            "not_for": "只是在讨论语音相关话题、转述别人发的语音、描述一段录音，或者要求用文字回复",
                            "examples": [
                                "你发段语音我听听",
                                "我想听听你的声音",
                                "用语音回答我",
                                "唱首歌给我听",
                                "你能不能语音说",
                            ],
                        },
                        "text": {
                            "what": "普通聊天、提问、指令、追问；或者明确要求打字、用文字回复；或者只是顺带提到「语音/声音/音频」但并不是要对方发语音",
                            "not_for": "任何要求对方发语音、想听声音、念或唱的表达",
                            "examples": [
                                "今天天气怎么样",
                                "别发语音，打字说",
                                "这个语音识别模型效果不错",
                                "帮我把这段录音转成文字",
                                "刚才那条语音是谁发的",
                            ],
                        },
                    },
                }
            },
        }

        model_name = str(self.config.get("jev_model") or "").strip() or "jev-latest"
        payload["model"] = model_name

        data = await self._post_systemone(url, api_key, payload, timeout)
        if data is None:
            return None

        # 调用失败或响应畸形都不能当成 NO；返回 None 让调用方回退对话模型。
        answers = data.get("answers") if isinstance(data, dict) else None
        answer = answers.get("reply_mode") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            logger.warning("[voice_intent] Jev 响应缺少有效的 Choice 答案，回退到对话模型")
            return None

        choice = str(answer.get("choice") or "").strip().lower()
        raw_confidence = answer.get("confidence")
        if (
            choice not in ("voice", "text")
            or isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
        ):
            logger.warning("[voice_intent] Jev 的 choice / confidence 无效，回退到对话模型")
            return None

        confidence = float(raw_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            logger.warning("[voice_intent] Jev 的 confidence 超出 0~1，回退到对话模型")
            return None

        want = choice == "voice" and confidence >= threshold

        if bool(self.config.get("debug", False)):
            cost = time.monotonic() - started
            probs = answer.get("probabilities") or {}
            logger.info(
                f"[voice_intent] Jev({data.get('model') or model_name}) 判断={choice!r} "
                f"置信度={confidence:.2f} 概率={probs} → {'语音' if want else '文字'}"
                f"（{cost:.2f}s / {user_text[:60]}）"
            )
        return want

    async def _post_systemone(
        self, url: str, api_key: str, payload: dict, timeout: float
    ) -> dict | None:
        """POST /v1/systemone，临时错误按退避策略重试一次，其余错误直接放弃。

        429 / 529 按官方要求指数退避，并遵守 Retry-After；连接错误和常见 5xx
        也重试一次。若服务端要求的等待超过本次判断超时，则直接回退对话模型。

        发送前先剥掉 key 的首尾空白；带内部空格、控制字符或非 ASCII 的 key
        不能组成合法的 Bearer 头，在发出去之前直接拒绝。
        """
        key = api_key.strip()
        if not key or not key.isascii() or any(c.isspace() or ord(c) < 32 for c in key):
            logger.warning("[voice_intent] TypeSafe API Key 格式不合法，已跳过 Jev")
            return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(0.3)
                    continue
                logger.warning(f"[voice_intent] Jev 请求失败，回退到对话模型：{e}")
                return None

            status = resp.status_code
            if status < 400:
                try:
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"[voice_intent] Jev 返回的不是 JSON：{e}")
                    return None
                if not isinstance(data, dict):
                    logger.warning("[voice_intent] Jev 返回的 JSON 顶层不是对象")
                    return None
                return data

            detail = ""
            try:
                detail = str(resp.text)[:300]
            except Exception:
                pass

            # 官方明确要求 429 / 529 退避重试；常见临时 5xx 也按同样策略处理。
            if status in (429, 500, 502, 503, 504, 529) and attempt == 0:
                delay = _retry_delay(resp, attempt)
                if delay > timeout:
                    logger.warning(
                        f"[voice_intent] Jev 返回 HTTP {status}，Retry-After "
                        f"等待 {delay:.2f}s 超过判断超时，直接回退到对话模型"
                    )
                    return None
                await asyncio.sleep(delay)
                continue

            if status in (401, 403):
                logger.warning(
                    "[voice_intent] TypeSafe API Key 无效或没权限"
                    f"（HTTP {status}），请检查 jev_api_key"
                )
            elif status == 404:
                logger.warning(
                    f"[voice_intent] Jev 接口或模型不存在（404），检查 jev_base_url / jev_model：{detail}"
                )
            elif status == 422:
                logger.warning(f"[voice_intent] Jev 认为请求体不合法（422）：{detail}")
            else:
                logger.warning(f"[voice_intent] Jev 返回 HTTP {status}，回退到对话模型：{detail}")
            return None

        return None

    async def _judge(self, event: AstrMessageEvent, req) -> bool:
        started = time.monotonic()

        user_text = ""
        try:
            user_text = str(getattr(req, "prompt", None) or "").strip()
        except Exception:
            user_text = ""
        if not user_text:
            user_text = (event.message_str or "").strip()
        if not user_text:
            return False

        # 优先走 Jev；没配 key 或调用失败时自动回退到对话模型
        if bool(self.config.get("use_jev", True)) and self._jev_key():
            want = await self._judge_jev(user_text, req, started)
            if want is not None:
                return want

        return await self._judge_llm(event, user_text, req, started)

    async def _judge_llm(
        self, event: AstrMessageEvent, user_text: str, req, started: float
    ) -> bool:
        provider = self._resolve_provider(event.unified_msg_origin)
        if provider is None:
            return False

        system_prompt = (
            str(self.config.get("intent_prompt") or "").strip() or DEFAULT_INTENT_PROMPT
        )
        prompt = self._build_judge_prompt(user_text, req)

        timeout = float(self.config.get("judge_timeout", 8) or 8)
        try:
            resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    contexts=[],
                    system_prompt=system_prompt,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[voice_intent] 意图判断超时（{timeout}s），按文字回复")
            return False
        except Exception as e:
            logger.warning(f"[voice_intent] 意图判断失败：{e}")
            return False

        answer = str(getattr(resp, "completion_text", "") or "").strip()
        reasoning = str(getattr(resp, "reasoning_content", "") or "")
        want = _parse_decision(answer, reasoning)
        if not answer and bool(self.config.get("debug", False)):
            logger.warning(
                "[voice_intent] 判断模型没有返回正文，可能是 reasoning 吃掉了 token；"
                "请在模型配置里设置 reasoning_effort=none"
            )
        if bool(self.config.get("debug", False)):
            cost = time.monotonic() - started
            logger.info(
                f"[voice_intent] 对话模型判断={answer[:40]!r} → {'语音' if want else '文字'} "
                f"（{cost:.2f}s / {user_text[:60]}）"
            )
        return want

    # ---------------- 发送前：按判断结果转语音 ----------------

    @filter.on_decorating_result()
    async def _on_decorating_result(self, event: AstrMessageEvent):
        try:
            self._manage_builtin_tts(event.unified_msg_origin)
        except Exception:
            pass

        try:
            await self._maybe_voice(event)
        except Exception:
            logger.error("[voice_intent] 处理语音回复失败：\n" + traceback.format_exc())

    async def _maybe_voice(self, event: AstrMessageEvent) -> None:
        if not bool(self.config.get("enabled", True)):
            return

        # 模型已经调过 send_voice：语音发出去了，这里只负责把重复的正文丢掉
        if event.get_extra(_VOICE_SENT_KEY):
            event.set_extra(_VOICE_SENT_KEY, None)
            sent_result = event.get_result()
            if sent_result is not None and sent_result.chain:
                logger.info("[voice_intent] 本轮已用 send_voice 发过语音，丢弃重复正文")
                sent_result.chain = []
            return

        force_voice = bool(event.get_extra(_FORCE_VOICE_KEY))
        task = event.get_extra(_TASK_KEY)
        if task is not None:
            event.set_extra(_TASK_KEY, None)

        result = event.get_result()
        if result is None or not result.chain:
            return
        if not _is_llm_result(result):
            return

        if force_voice:
            # 模型主动尝试发 record 本身就是明确的语音意图。无需再等 Jev，
            # 否则「你再看看」这类承接收尾句会被判成文字，抵消刚才的拦截。
            if task is not None and not task.done():
                task.cancel()
        else:
            if task is None:
                return
            if not await self._await_judge(task):
                return

        if bool(self.config.get("only_private", False)):
            try:
                if not event.is_private_chat():
                    return
            except Exception:
                pass

        if SessionServiceManager is not None:
            try:
                if not await SessionServiceManager.should_process_tts_request(event):
                    return
            except Exception:
                pass

        try:
            tts = await self._resolve_tts_provider(event)
        except Exception as e:
            logger.warning(f"[voice_intent] 获取 TTS 提供商失败：{e}")
            return
        if tts is None:
            logger.warning("[voice_intent] 需要发语音，但没有可用的 TTS 提供商")
            return

        plain_texts = [
            str(getattr(c, "text", "") or "")
            for c in result.chain
            if isinstance(c, Comp.Plain)
        ]
        full_text = "".join(plain_texts).strip()
        if not full_text:
            return

        max_chars = int(self.config.get("max_chars", 300) or 0)
        if max_chars > 0 and len(full_text) > max_chars:
            logger.info(
                f"[voice_intent] 回复 {len(full_text)} 字，超过上限 {max_chars}，保持文字"
            )
            return

        await self._convert_to_voice(event, result, tts)

    async def _resolve_tts_provider(self, event: AstrMessageEvent):
        """拿到可用的 TTS 提供商。

        插件会保持 provider_tts_settings.enable 为开启状态，正常情况下
        get_using_tts_provider_async 直接可用；后面的手动选择用于兼容
        旧版 AstrBot 或会话级配置异常。
        """
        try:
            provider = await self.context.get_using_tts_provider_async(
                event.unified_msg_origin
            )
            if provider is not None:
                return provider
        except Exception:
            pass

        try:
            providers = self.context.get_all_tts_providers() or []
        except Exception:
            providers = []
        if not providers:
            return None

        provider_id = ""
        try:
            conf = self.context.get_config(event.unified_msg_origin) or {}
            provider_id = str(
                (conf.get("provider_tts_settings") or {}).get("provider_id") or ""
            )
        except Exception:
            pass

        if provider_id:
            for candidate in providers:
                cfg = getattr(candidate, "provider_config", None) or {}
                if str(cfg.get("id") or "") == provider_id:
                    return candidate

        return providers[0]

    async def _await_judge(self, task: asyncio.Task) -> bool:
        try:
            # Jev 可能重试一次并回退对话模型，兜底上限必须跟随配置，不能硬编码 15 秒。
            configured = float(self.config.get("judge_timeout", 8) or 8)
            timeout = max(15.0, configured * 3 + 2)
            return bool(await asyncio.wait_for(asyncio.shield(task), timeout=timeout))
        except Exception:
            return False

    async def _convert_to_voice(self, event: AstrMessageEvent, result, tts) -> None:
        dual_output = bool(self.config.get("dual_output", False))

        use_file_service = False
        callback_api_base = ""
        try:
            conf = self.context.get_config(event.unified_msg_origin) or {}
            tts_settings = conf.get("provider_tts_settings") or {}
            use_file_service = bool(tts_settings.get("use_file_service", False))
            callback_api_base = str(conf.get("callback_api_base") or "")
        except Exception:
            pass

        new_chain: list = []
        converted = 0
        scrub = bool(self.config.get("scrub_voice_refusal", True))
        for comp in result.chain:
            if isinstance(comp, Comp.Plain) and len(str(comp.text or "")) > 1:
                text = str(comp.text)
                if scrub:
                    try:
                        text = self._scrub_voice_refusal(text)
                    except Exception:
                        text = str(comp.text)
                try:
                    audio_path = await tts.get_audio(text)
                except Exception as e:
                    logger.error(f"[voice_intent] 语音合成失败，保留文字：{e}")
                    new_chain.append(comp)
                    continue

                if not audio_path:
                    logger.error("[voice_intent] 语音合成没有返回文件，保留文字")
                    new_chain.append(comp)
                    continue

                url = await self._register_audio_url(event, audio_path)

                new_chain.append(
                    Comp.Record(
                        file=url or audio_path,
                        url=url or audio_path,
                        text=text,
                    )
                )
                converted += 1
                if dual_output:
                    new_chain.append(comp)
            else:
                new_chain.append(comp)

        if converted:
            result.chain = new_chain
            logger.info(f"[voice_intent] 已转为语音回复（{converted} 段）")
        else:
            logger.info("[voice_intent] 没有可转换的文本段，保持文字回复")

    # ---------------- 状态指令 ----------------

    @filter.command("voice")
    async def voice_status(self, event: AstrMessageEvent):
        """查看语音意图回复的当前状态。"""
        umo = event.unified_msg_origin
        lines = []

        enabled = bool(self.config.get("enabled", True))
        lines.append(f"插件：{'启用' if enabled else '停用'}")

        tts_state = "未接管"
        try:
            conf = self.context.get_config(umo) or {}
            tts_settings = conf.get("provider_tts_settings") or {}
            if tts_settings.get("enable"):
                if enabled and self._stage_module is not None:
                    tts_state = "开启（概率触发已屏蔽）"
                else:
                    tts_state = "开启"
            else:
                tts_state = "关闭"
        except Exception:
            pass
        lines.append(f"内置 TTS 状态：{tts_state}")

        use_jev = bool(self.config.get("use_jev", True))
        has_key = bool(self._jev_key())
        if use_jev and has_key:
            model_name = str(self.config.get("jev_model") or "jev-latest").strip()
            threshold = float(self.config.get("jev_confidence", 0.6) or 0)
            lines.append(f"判断方式：Jev（{model_name}，置信度阈值 {threshold:g}）")
        elif use_jev:
            lines.append("判断方式：对话模型（Jev 未填 API Key）")
        else:
            provider = self._resolve_provider(umo)
            if provider is None:
                lines.append("判断方式：对话模型（无可用模型）")
            else:
                cfg = getattr(provider, "provider_config", None) or {}
                pid = str(cfg.get("id") or cfg.get("model") or "未知")
                lines.append(f"判断方式：对话模型（{pid}）")

        try:
            tts = await self._resolve_tts_provider(event)
            if tts is None:
                lines.append("语音提供商：无")
            else:
                tcfg = getattr(tts, "provider_config", None) or {}
                lines.append(f"语音提供商：{tcfg.get('id') or '未知'}")
        except Exception:
            lines.append("语音提供商：读取失败")

        try:
            if SessionServiceManager is not None:
                ok = await SessionServiceManager.should_process_tts_request(event)
                lines.append(f"本会话语音开关：{'开' if ok else '关（插件不会发语音）'}")
        except Exception:
            pass

        use_tool = bool(self.config.get("use_voice_tool", True))
        tell = bool(self.config.get("tell_model_voice_auto", True))
        strip = bool(self.config.get("strip_voice_tool", True))
        if use_tool:
            model_side = "send_voice 工具（判为语音时提供给模型）"
        else:
            model_side = "已告知语音自动" if tell else "未告知"
        lines.append(
            f"模型侧：{model_side}"
            f" / {'已禁止工具发语音' if strip else '未限制工具发语音'}"
        )

        yield event.plain_result("\n".join(lines))

    @filter.command("voicetest")
    async def voice_test(self, event: AstrMessageEvent):
        """用固定的一句话跑一次真实判断，验证当前实际生效的判断链路。"""
        text = "你发段语音我听听"
        lines = [f"测试句：{text}", "预期结果：语音", ""]

        if not bool(self.config.get("enabled", True)):
            lines.append("插件当前是停用状态")
            yield event.plain_result("\n".join(lines))
            return

        # 与正式消息一致：Jev 成功就直接用；只有未配置或调用失败才回退对话模型。
        use_jev = bool(self.config.get("use_jev", True))
        if use_jev and self._jev_key():
            started = time.monotonic()
            want = await self._judge_jev(text, None, started)
            cost = time.monotonic() - started
            if want is not None:
                mark = "✓" if want else "✗ 与预期不符"
                lines.append(
                    f"Jev：判为{'语音' if want else '文字'}（{cost:.2f}s）{mark}"
                )
                yield event.plain_result("\n".join(lines))
                return
            lines.append(f"Jev：调用失败（{cost:.2f}s），回退对话模型")
        elif use_jev:
            lines.append("Jev：未配置 API Key，回退对话模型")
        else:
            lines.append("Jev：已关闭，使用对话模型")

        provider = self._resolve_provider(event.unified_msg_origin)
        if provider is None:
            lines.append("对话模型：没有可用模型")
        else:
            cfg = getattr(provider, "provider_config", None) or {}
            pid = str(cfg.get("id") or cfg.get("model") or "未知")
            started = time.monotonic()
            want = await self._judge_llm(event, text, None, started)
            cost = time.monotonic() - started
            mark = "✓" if want else "✗ 与预期不符"
            lines.append(
                f"对话模型（{pid}）：判为{'语音' if want else '文字'}（{cost:.2f}s）{mark}"
            )

        yield event.plain_result("\n".join(lines))
