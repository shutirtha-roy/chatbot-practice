#import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_initial_message():
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful AI Tutor, who answers brief questions about AI."
        },
        {
            "role": "user", 
            "content": "I want to learn AI"
        },
        {
            "role": "assistant", 
            "content": "That's awesome, what do you want to know about AI"
        }

   
    ]

    return messages

def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model = model,
        messages = messages
    )

    return response.choices[0].message.content

def update_chat(messages, role, content):
    messages.append(
        { 
            "role": role,
            "content": content
        }
    )

    return messages