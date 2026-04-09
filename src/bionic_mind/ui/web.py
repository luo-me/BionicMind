from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


async def run_web(config_path: str = "config.yaml", host: str = "0.0.0.0", port: int = 7860) -> None:
    try:
        import gradio as gr
    except ImportError:
        logger.error("gradio not installed. Run: pip install gradio")
        return

    from bionic_mind.core.mind import BionicMind

    mind = BionicMind(config_path=config_path)

    with gr.Blocks(title="BionicMind - 仿生意识系统") as demo:
        gr.Markdown("# 🧠 BionicMind - 仿生意识系统")
        gr.Markdown("具备记忆、情绪和内驱力的AI意识框架")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg_input = gr.Textbox(
                    placeholder="输入消息...",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

            with gr.Column(scale=1):
                gr.Markdown("### 内部状态")
                emotion_display = gr.JSON(label="情绪状态", value={})
                drive_display = gr.JSON(label="内驱力", value={})
                memory_display = gr.JSON(label="记忆统计", value={})
                with gr.Row():
                    feedback_pos = gr.Button("👍 正向反馈", size="sm")
                    feedback_neg = gr.Button("👎 负向反馈", size="sm")

        async def respond(message: str, history: list):
            if not message.strip():
                return "", history, {}, {}, {}

            result = await mind.run_cycle(message)
            if result is None:
                return "", history, {}, {}, {}

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result.output})

            emotion_data = result.emotion.to_dict()
            emotion_data["mode"] = result.emotion.mode.value
            drive_data = result.drives.to_dict()
            dominant_name, dominant_value = result.drives.dominant()
            drive_data["dominant"] = f"{dominant_name}({dominant_value:.2f})"
            memory_data = mind.memory.get_stats()
            memory_data["working_memory"] = len(mind.working_memory)

            return "", history, emotion_data, drive_data, memory_data

        async def give_positive_feedback():
            mind.drives.update(social_feedback=0.5)
            return mind.emotion.to_dict(), mind.drives.to_dict(), mind.memory.get_stats()

        async def give_negative_feedback():
            mind.drives.update(social_feedback=-0.5)
            return mind.emotion.to_dict(), mind.drives.to_dict(), mind.memory.get_stats()

        async def clear_chat():
            mind.working_memory.clear()
            return [], {}, {}, {}

        send_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, emotion_display, drive_display, memory_display],
        )
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, emotion_display, drive_display, memory_display],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, emotion_display, drive_display, memory_display])
        feedback_pos.click(give_positive_feedback, outputs=[emotion_display, drive_display, memory_display])
        feedback_neg.click(give_negative_feedback, outputs=[emotion_display, drive_display, memory_display])

    logger.info(f"Starting web UI on {host}:{port}")
    demo.launch(server_name=host, server_port=port, theme=gr.themes.Soft())
