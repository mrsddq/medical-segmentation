from __future__ import annotations

import gradio as gr


def explain_status() -> str:
    return (
        "Upload/inference UI scaffold is ready. Train a checkpoint with scripts/train.py, "
        "then wire the checkpoint path into scripts/infer.py before deploying."
    )


demo = gr.Interface(fn=explain_status, inputs=None, outputs="text", title="Cardiac MRI Segmentation")


if __name__ == "__main__":
    demo.launch()
