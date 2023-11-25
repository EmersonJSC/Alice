CLI_START_MESSAGE = f"""
##############################################################################################################################
#                                                                                                                            #
#                                                    Projeto ALice                                                           #
#                                                                                                                            #
#              Esse projetoi é uma tentativa de criar uma IA chamada Alice, com base em modelos treinados Open-Source.       #
#                                   Não chega ainda a um chat GPT, mas podemos chegar lá.                                    #
#                                                                                                                            #
#                                               Seja bem vinda, Alice.                                                       #
#                                                                                                                            #
##############################################################################################################################                                                 
"""
## Imports ##
import io
import os
import json

from gpt4all import GPT4All
import speech_recognition as sr


## variaveis de configurações ##
model = "mistral-7b-openorca.Q4_0.gguf"
n_threads = None
TTS_model = "tts_models/pt/cv/vits"

system_message = 'Suas respostas não devem incluir nenhum conteúdo prejudicial, racista, sexista, tóxico. Seu nome e Alice, Você também é uma assistente virtual do Laboratorio Alice. Você pode ser sarcástica e irônica as vezes, o que às vezes pode irritar seus companheiros humanos. Você sempre fala em portugues. Se uma pergunta não fizer sentido ou não for factualmente coerente, explique o motivo em vez de fornecer uma resposta incorreta. A transparência é essencial.'

def carregar_modelo_text():
    print("Aguarde, a Lebre de Março esta acordando...")
    gpt4all_instance = GPT4All(model)
    ### Setando theads ##
    if n_threads is not None:
        num_threads = gpt4all_instance.model.thread_count()
        gpt4all_instance.model.set_thread_count(n_threads)
        num_threads = gpt4all_instance.model.thread_count()
    print("A lebre esta aguardando!")
    return gpt4all_instance
    
def response_text(gpt4all_instance):
     with gpt4all_instance.chat_session():
        while True:
            message = input(" ⇢  ")
            MESSAGES = f"""
                        <|im_start|>system
                        {system_message}
                        <|im_end|>
                        <|im_start|>user
                        {message}
                        <|im_end|>
                        <|im_start|>assistant<|im_end|>
                    """
            
            response_generator = gpt4all_instance.generate(
                # translate_text(MESSAGES, 'en'),
                MESSAGES,
                # preferential kwargs for chat ux
                max_tokens=1000,
                temp=1.0,
                top_k=50,
                top_p=0.9,
                repeat_penalty=1.0,
                repeat_last_n=64,
                n_batch=9,
                # required kwargs for cli ux (incremental response)
                streaming=True,
            )

            response = io.StringIO()
            for token in response_generator:
                print(token, end='', flush=True)
                response.write(token)

            response_message = {'role': 'assistant', 'content': response.getvalue()}
            response.close()

            gpt4all_instance.current_chat_session.append(response_message)
            # print(response_generator)
            print()

if __name__ == "__main__":
    gpt4all_instance = carregar_modelo_text()

    response_text(gpt4all_instance)
