"""Gradio UI, upload an image and get a caption back. Mounted inside app/main.py."""
import gradio as gr

from app.inference import caption_pil, MODE


def _predict(image):
    if image is None:
        return "Please upload an image."
    return caption_pil(image)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Image Caption Generator") as demo:
        gr.Markdown(
            "# Image Caption Generator\n"
            "Upload an image and get a generated caption. CNN encoder (transfer "
            "learning) plus attention plus an RNN decoder, trained on Flickr8k.\n\n"
            f"Active model mode: `{MODE}`"
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an image")
            caption_output = gr.Textbox(label="Generated caption", lines=3)
        submit_btn = gr.Button("Generate caption", variant="primary")
        submit_btn.click(fn=_predict, inputs=image_input, outputs=caption_output)
        image_input.change(fn=_predict, inputs=image_input, outputs=caption_output)

        gr.Markdown(
            "Same container also serves the REST API, POST /caption with a "
            "multipart image upload, or check /docs for the interactive docs."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
