import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.cass import create_ass
from main.app.core.ui import shutil_move, change_audios_choices
from main.app.variables import translations, file_types, configs, paths_for_files

def create_ass_tab():
    with gr.Row():
        gr.Markdown("### Tạo file ASS với word timestamps")
    with gr.Row():
        with gr.Column():
            ass_content = gr.Textbox(label="Nội dung ASS", value="", lines=9, max_lines=9, interactive=False)
        with gr.Column():
            model_size = gr.Radio(label="Kích thước mô hình Whisper", info="Chọn mô hình Whisper để sử dụng", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], value="medium", interactive=True)
            style_name = gr.Textbox(label="Tên Style", value="Dòng trên", info="Tên style cho dialogue (ví dụ: Dòng trên, Dòng dưới)", interactive=True)
    with gr.Row():
        convert_button = gr.Button("Tạo file ASS", variant="primary")
    with gr.Row():
        with gr.Accordion("Input/Output", open=False):
            with gr.Column():
                input_audio = gr.Dropdown(label="Đường dẫn audio", value="", choices=paths_for_files, info="Chọn file audio", allow_custom_value=True, interactive=True)
                output_file = gr.Textbox(label="File ASS output", value="ass/output.ass", placeholder="ass/output.ass", interactive=True)
            with gr.Column():
                refresh = gr.Button("Làm mới")
            with gr.Row():
                input_file = gr.Files(label="Kéo thả file audio", file_types=file_types)
    with gr.Row():
        play_audio = gr.Audio(show_download_button=True, interactive=False, label="Audio input")
    with gr.Row():
        output_ass = gr.File(label="File ASS output", file_types=[".ass"], interactive=False, visible=False)
    with gr.Row():
        input_file.upload(fn=lambda audio_in: [shutil_move(audio.name, configs["audios_path"]) for audio in audio_in][0], inputs=[input_file], outputs=[input_audio])
        input_audio.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio], outputs=[play_audio])
        refresh.click(fn=change_audios_choices, inputs=[input_audio], outputs=[input_audio])
    with gr.Row():
        convert_button.click(
            fn=create_ass,
            inputs=[
                model_size, 
                input_audio, 
                output_file, 
                gr.Checkbox(value=True, visible=False),  # word_timestamps - luôn True
                style_name
            ],
            outputs=[
                output_ass,
                ass_content
            ],
            api_name="create_ass"
        )

