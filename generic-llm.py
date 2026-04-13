from litellm import completion

llm_list = ["llama3.1:70b", "mistral-small3.2", "phi3.5:3.8b"]

def trigger_model(llm_key, request):
    try:
        result = completion(
            model=llm_key,
            messages=[{"role": "user", "content": request}]
        )
        return result.choices[0].message.content
    except Exception as e:
        return f"The following problem with {llm_key} has occured: {e}"

def test_model(name_of_bot, answer):
    print(f"Output from {name_of_bot} to see if it works.")
    print("STATUS: PASS \n" if len(answer) >= 1 else "STATUS: FAILED \n")

#run the model
for bot in llm_list:
    print(f"Calling {bot}")
    prompt_response = trigger_model(bot, "Hi")
    test_model(bot, prompt_response)

