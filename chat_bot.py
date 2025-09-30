# chat_bot.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1️⃣ Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./helpdesk-bot")
model = AutoModelForCausalLM.from_pretrained("./helpdesk-bot")

# 2️⃣ Create text-generation pipeline
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# 3️⃣ Interactive chat loop
chat_history = ""
print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bye!")
        break

    # Append current user input to chat history
    chat_history += f"User: {user_input}\nBot:"

    # Generate response
    response = chat(chat_history, max_new_tokens=50)[0]['generated_text']

    # Print bot reply
    print("Bot:", response.strip())

    # Update chat history
    chat_history += f" {response.strip()}\n"
