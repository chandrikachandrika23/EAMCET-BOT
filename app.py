import json
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Load FAQ model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load FAQs
with open("faqs.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Load College Data
college_data = pd.read_csv("colleges.csv")
# Rule-based responses (from your Streamlit code)
faq_responses = {
    "what is eamcet": "EAMCET stands for Engineering, Agriculture, and Medical Common Entrance Test. It is an entrance exam conducted for admissions into engineering and medical colleges in Telangana and Andhra Pradesh.",
    "who conducts eamcet": "EAMCET is conducted by Jawaharlal Nehru Technological University (JNTU) on behalf of the respective state councils.",
    "what is the eligibility for eamcet": "To appear for EAMCET, candidates must have completed their 10+2 education with Physics, Chemistry, and Mathematics/Biology as core subjects.",
    "how to apply for eamcet": "You can apply for EAMCET online through the official website by filling out the application form and paying the required fee.",
    "when is eamcet conducted": "EAMCET is usually conducted in the months of April or May. Exact dates vary each year and are announced on the official website.",
    "what is the syllabus for eamcet": "The EAMCET syllabus includes topics from Physics, Chemistry, and Mathematics/Biology from the 11th and 12th standard curriculum.",
    "what is the exam pattern for eamcet": "EAMCET is a computer-based test with multiple-choice questions. It consists of 160 questions: 80 from Mathematics, 40 from Physics, and 40 from Chemistry.",
    "how many marks is eamcet conducted for": "The EAMCET exam is conducted for a total of 160 marks, with no negative marking.",
    "how to download eamcet hall ticket": "You can download the hall ticket from the official EAMCET website by entering your registration details.",
    "what is the pass mark for eamcet": "The minimum qualifying marks for EAMCET is 40 out of 160 for general category candidates. There is no minimum qualifying mark for SC/ST candidates.",
    "how is the eamcet rank calculated": "The EAMCET rank is calculated based on 75% of the EAMCET score and 25% of the Intermediate Board marks in relevant subjects.",
    "when will eamcet results be declared": "EAMCET results are typically declared within a few weeks after the exam. The exact date is announced on the official website.",
    "how to check eamcet results": "You can check your EAMCET results by visiting the official website and entering your hall ticket number.",
    "how to download eamcet rank card": "The EAMCET rank card can be downloaded from the official website by entering your registration details.",
    "what is the counselling process for eamcet": "EAMCET counselling includes registration, document verification, choice filling, seat allotment, and final admission to colleges.",
    "what are the documents required for eamcet counselling": "Documents required include EAMCET hall ticket, rank card, SSC certificate, Intermediate marks memo, caste certificate (if applicable), income certificate, and residence proof.",
    "bye": "Goodbye! All the best for your EAMCET preparation and admission process!"
}

# Modified chatbot function
def chatbot(query):
    query_clean = query.lower().strip()
    
    # Rule-based check first
    if query_clean in faq_responses:
        return faq_responses[query_clean]
    
    # If not found, fallback to semantic search
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
    best_match = scores.argmax().item()
    return answers[best_match]

# # FAQ Chatbot
# def chatbot(query):
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]
#     best_match = scores.argmax().item()
#     return answers[best_match]

# College Predictor
def college_predictor(rank, category):
    results = college_data[college_data["category"] == category]
    results = results[results["cutoff_rank"] >= rank]
    
    if results.empty:
        return "❌ No colleges found for your rank in this category."
    
    suggestions = results.sort_values("cutoff_rank").head(5)
    output = "🎓 Based on your rank, you may get:\n\n"
    for _, row in suggestions.iterrows():
        output += f"- {row['college']} ({row['branch']}) [Cutoff: {row['cutoff_rank']}]\n"
    return output

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 EAMCET Chatbot + 🎓 College Predictor")

    with gr.Tab("Chatbot"):
        gr.Markdown("Ask me anything about EAMCET (exam pattern, eligibility, syllabus, etc.)")
        chatbox = gr.Chatbot()
        msg = gr.Textbox(label="Your Question")
        def respond(user_message, chat_history):
            bot_message = chatbot(user_message)
            chat_history.append((user_message, bot_message))
            return "", chat_history
        msg.submit(respond, [msg, chatbox], [msg, chatbox])

    with gr.Tab("College Predictor"):
        gr.Markdown("Enter your rank and category to get possible colleges")
        rank_input = gr.Number(label="Enter Your Rank")
        category_input = gr.Dropdown(choices=["OC", "BC", "SC", "ST"], label="Select Category")
        predict_btn = gr.Button("Predict Colleges")
        output = gr.Textbox(label="Predicted Colleges")

        predict_btn.click(fn=college_predictor, inputs=[rank_input, category_input], outputs=output)

demo.launch()
