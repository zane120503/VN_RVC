import gradio as gr
from main.app.tabs.automation.child.automation import automation_tab as child_automation_tab

def automation_tab():
    with gr.Tab("✨ Tự Động Hóa (Auto)", id="automation_tab"):
        child_automation_tab()
