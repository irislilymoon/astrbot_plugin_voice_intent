# 语音意图回复

[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.24.2-blue)](https://astrbot.app)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

AstrBot 自带的语音回复是「按概率掷骰子」：`provider_tts_settings.trigger_probability` 设成 0.1，就十条里随机一条变语音，跟用户在说什么没关系。

这个插件把那个骰子拿走，换成一次意图判断：

- 用户说「你发段语音我听听」「我想听听你的声音」→ 判断要语音 → 回复转语音
- 用户只是正常聊天、问问题、提到「语音」二字但不是在要求发语音 → 正常文字

## 安装

从插件市场搜索「语音意图回复」安装，或手动放置：

```bash
cd /path/to/astrbot/data/plugins
git clone https://github.com/irislilymoon/astrbot_plugin_voice_intent
```

重启 AstrBot，在 WebUI 的插件配置里打开「启用语音意图回复」。
需要先在「服务提供商」中配置好至少一个 TTS 提供商，插件本身不带 TTS 能力。

## 判断方式：Jev 或对话模型

默认走 **Jev**（TypeSafe 的 System One 模型）。它不是聊天模型，而是「给一个 state、问几个结构化问题、拿回选项和置信度」，正好适合这种二分类判断：

```json
POST https://api.typesafe.ai/v1/systemone
{
  "state": {
    "recent_dialog": ["用户: ……", "助手: ……"],
    "latest_user_message": "本轮用户消息"
  },
  "model": "jev-latest",
  "questions": {
    "reply_mode": {
      "type": "choice",
      "instructions": {
        "question": "`latest_user_message` 是否在要求对方用语音的形式回复？",
        "focus": "`recent_dialog` 仅用于理解省略和指代"
      },
      "criteria": {
        "voice": { "what": "要求发语音、想听声音、念或唱…", "not_for": "只是讨论语音话题…" },
        "text":  { "what": "普通聊天、提问、明确要求打字…",  "not_for": "要求发语音…" }
      }
    }
  }
}
```

返回 `answers.reply_mode.choice`（`voice` / `text`）和 `confidence`。只有 `choice == voice` 且 `confidence >= 阈值` 才发语音。二分类下 `confidence = |2 × 最高概率 - 1|`，所以默认阈值 0.6 实际要求 `voice` 概率至少为 0.8，不是 0.6。

没填 API Key、或者 Jev 接口出错时，自动回退到 AstrBot 里的对话模型（原先那套 YES/NO 提示词判断），不会因为一个服务挂掉就整条链路失效。

官方说明 Jev 支持中文等 CJK 输入，但英文准确率更高。中文分类效果和阈值仍需用真实样本验证，不能仅凭接口结构正确就保证判断准确率。

## 内置 TTS 状态与停用恢复

插件启用时把运行期内存中的 `provider_tts_settings.enable` 保持为 `True`，让依赖 TTS 状态的功能认为语音回复可用；同时临时屏蔽 `ResultDecorateStage` 里的随机数判断，使内置概率触发永远不会命中。实际是否转语音完全交给意图判断。

插件停用时恢复 `enable` 为本插件接管前的值：原本开启就恢复开启，原本关闭就恢复关闭；同时恢复被屏蔽的概率触发。磁盘上的 `cmd_config.json` 不会被改写，重启后按原配置启动。

## 模型侧：让模型知道语音是自动的

语音是在发送阶段由 TTS 把文字转出来的，模型看不到这个转换，所以它并不知道自己「能发语音」。在智能体模式下，模型会主动调用 `send_message_to_user` 尝试自己发 `record`，沙箱未启用时收到报错，随后就把这条报错当真，回一句「我发不了语音 / 沙箱没开」。这段自我否定的文字又会经 TTS 变成语音发出去，形成闭环。

插件在 `on_llm_request` 里做两件事切断这个闭环：

1. **追加语音声明**（`tell_model_voice_auto`）：在系统提示末尾说明语音由发送阶段自动合成，模型不需要也无法自己发音频，禁止声称「没有 TTS 接口」「沙箱未启用」。
2. **在执行层拦截 record**（`strip_voice_tool`）：先把 `send_message_to_user` 的 `record` 从枚举和描述里移除，再用请求级代理包住真实工具。模型即使无视枚举、仍传 `record`，也不会访问沙箱或收到「沙箱未启用」错误；插件会把本轮标记为必须语音，让最终文字回复经 TTS 发出。普通图片、视频、文件、@ 调用仍交给原工具，原始工具对象不被修改。

两项都可在配置里单独关闭。

## 配置项

| 配置 | 说明 |
| --- | --- |
| 启用语音意图回复 | 总开关 |
| 保持 TTS 开启但屏蔽概率触发 | 默认开。关掉则内置概率语音和意图判断会叠加 |
| 告诉模型「语音由系统自动转换」 | 默认开。关掉则模型可能继续声称自己发不了语音 |
| 禁止模型用工具自己发语音 | 默认开。同时修改工具说明并在执行层拦截 `record`；违规调用会改走最终回复自动转语音 |
| 用 Jev 做结构化判断 | 默认开。关掉则只用对话模型 |
| TypeSafe API Key | `console.typesafe.ai/keys` 创建。留空自动回退对话模型 |
| Jev 模型 | 默认 `jev-latest` |
| Jev API 地址 | 默认 `https://api.typesafe.ai`；也支持带 `/v1` 的地址或完整 `/v1/systemone` 端点 |
| 判为语音的最低置信度 | 默认 0.6。调低更容易发语音，调高更保守 |
| 对话模型 | 下拉框，选 AstrBot 里的 Provider；留空 = 当前会话模型。Jev 关闭或不可用时用它 |
| 对话模型判断用的系统提示词 | 留空用内置规则；自定义时须保证模型只输出 YES / NO |
| 意图判断超时 | 默认 8 秒，超时按「不需要语音」处理；响应较慢的第三方镜像建议设为 20～30 秒 |
| 判断时附带最近几条对话 | 默认开，让「再来一段」这种省略句也能判对 |
| 转语音的最大字数 | 默认 300，超出直接发文字 |
| 语音 + 文字同时发 | 默认关，只发语音 |
| 只在私聊生效 | 默认关 |
| 输出调试日志 | 打开后日志里有判断结果、置信度和耗时 |

## 指令

`/voice` —— 查看当前状态：插件是否启用、内置 TTS 状态与概率屏蔽、判断走 Jev 还是对话模型、TTS 提供商、本会话语音开关、模型侧两项处理是否生效。

`/voicetest` —— 用固定的一句话（「你发段语音我听听」，预期结果是语音）跑一次正式判断链路：优先 Jev，Jev 未配置或调用失败才回退对话模型，不会重复判断。打印实际使用的判断方式和耗时。

## 错误处理

- `401 / 403`：API Key 无效或没权限，日志里直接点名
- `404`：接口或模型不存在，提示检查 `jev_base_url` / `jev_model`
- `422`：请求体不合法
- `429 / 529`：按官方要求指数退避后重试一次，并遵守 `Retry-After`；等待超过判断超时时直接回退
- `500 / 502 / 503 / 504`：按同样的退避策略重试一次，仍失败才回退
- 连接异常：重试一次
- HTTP 成功但响应不是合法 Choice 答案：按调用失败处理，回退对话模型，不误判为文字
- key 带内部空格、控制字符或非 ASCII：发请求之前就拒绝，避免无谓的网络往返

任何一条走不通都回退到对话模型，不会因为 TypeSafe 挂了就整条链路失效。

## 依赖与前提

- AstrBot 已在「服务提供商」里配好至少一个 TTS 提供商（本机是 `mimo_tts_clone`），插件本身不带 TTS 能力。
- 会话级语音开关（WebUI 里对单个会话关掉 TTS）优先于本插件：关掉后插件也不会发语音。

## 排查

打开「输出调试日志」后，日志里能看到判断依据：

```
[voice_intent] Jev(jev-1.14.0) 判断='voice' 置信度=0.94 概率={'voice': 0.97, 'text': 0.03} → 语音（0.41s / 你发段语音我听听）
[voice_intent] TypeSafe API Key 无效或没权限（HTTP 401），请检查 jev_api_key
```

判断偏保守就把「判为语音的最低置信度」调低到 0.4~0.5；误判太多就调高到 0.7 以上。
