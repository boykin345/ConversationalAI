from chatbot import Chatbot

def main():
    chatbot = Chatbot()
    print(chatbot.get_welcome_message())

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break

        response = chatbot.handle_user_input(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
