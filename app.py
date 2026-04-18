import gradio as gr
from engine import FraudSentinel, LogManager
from datetime import datetime

# Initialize backend components
sentinel = FraudSentinel()
db = LogManager()

def process(val, rets, age, delivery, refund, method, rating):
    # Data Mapping
    method_map = {"Card": 0, "Wallet": 1, "Crypto": 2}
    input_data = {
        "order_value": val, "return_count": rets, "account_age": age,
        "delivery_time": delivery, "refund_amount": refund,
        "payment_method": method_map[method], "user_rating": rating
    }
    
    # AI Analysis
    score = sentinel.predict(input_data)
    verdict = "🚨 REJECTED" if score > 0.75 else "⚠️ REVIEW" if score > 0.4 else "✅ APPROVED"
    
    # Persistence (Database Write)
    db.add_log({"Time": datetime.now(), "Amount": val, "Score": score, "Verdict": verdict})
    
    return {score: 1.0}, verdict

# UX Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ SENTINEL AI: TRANSACTION GUARD")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Transaction")
            val = gr.Number(label="Transaction Value ($)", value=250)
            refund = gr.Number(label="Refund Request ($)", value=0)
            method = gr.Dropdown(["Card", "Wallet", "Crypto"], label="Payment Method", value="Card")
            with gr.Accordion("Customer Metrics", open=False):
                age = gr.Slider(1, 1000, label="Account Age (Days)", value=90)
                rets = gr.Slider(0, 50, label="Past Returns", value=1)
                rating = gr.Slider(1, 5, label="Trust Rating", value=5)
            btn = gr.Button("Analyze Risk", variant="primary")
            
        with gr.Column():
            gr.Markdown("### AI Decision")
            prob_out = gr.Label(label="Fraud Confidence")
            verdict_out = gr.Textbox(label="Status")

    btn.click(process, inputs=[val, rets, age, gr.State(5), refund, method, rating], 
              outputs=[prob_out, verdict_out])

if __name__ == "__main__":
    demo.launch(share=True)
