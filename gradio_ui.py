import gradio as gr

from conversation import create_llm_conversation, handle_user_query

with gr.Blocks(
    css='footer {visibility: hidden}'
) as demo:
    chatbot = gr.Chatbot(label='Talk to the Douments', bubble_full_width=False)
    msg = gr.Textbox(label='Query', placeholder='Enter text and press enter')
    clear = gr.ClearButton([msg, chatbot], variant='stop')

    msg.submit(
        handle_user_query,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        create_llm_conversation,
        [chatbot],
        [chatbot]
    )

demo.queue()
