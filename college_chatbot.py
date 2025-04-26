import os
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = groq.Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_college_info(query):
    # System message to set context
    system_message = """You are a helpful assistant that provides information about colleges in Vellore, Tamil Nadu, India. 
    You should provide accurate information about admission processes, courses offered, campus facilities, and other relevant details."""
    
    try:
        # Create chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="llama-3.3-70b-versatile",  # Updated to the current supported version
            temperature=0.7,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    print("Welcome to the Vellore Colleges Chatbot!")
    print("Ask me anything about colleges in Vellore (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Vellore Colleges Chatbot!")
            break
            
        response = get_college_info(user_input)
        print("\nBot:", response)

if __name__ == "__main__":
    main()