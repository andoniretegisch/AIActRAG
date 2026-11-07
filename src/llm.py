import ollama

def generate_answer(prompt:str):

    response: ollama.ChatResponse = ollama.chat( model='gemma3:4b',messages=[{'role': 'user','content': prompt,},],)
    return response['message']['content']

def main():
    print(generate_answer("Hi! how are you?"))
    

if __name__ == "__main__":
    main()
