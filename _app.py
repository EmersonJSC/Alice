#const###################################################
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

# Data inicial da alice
# variaveis de configurações
model = "llama-2-7b-chat.Q4_K_M.gguf"
n_threads = 10

# imports
from gpt4all import GPT4All
import speech_recognition as sr
from googletrans import Translator
import io
import os
import json
from TTS.api import TTS
from gtts import gTTS
import pyttsx3
import pygame

# Iniciando o mic
r = sr.Recognizer()
mic = sr.Microphone()

def main():
                                                                                                         
    #inicializar a de texto
    gpt4all_instance = repl(model, n_threads)
    #Inicia ia de fala
    tts = carregar_modelo_fala()   
    # Loop principal
    # with gpt4all_instance.chat_session():
    with gpt4all_instance.chat_session():
        # MESSAGES = carregar_conversa()
        while True:
            message = input(" ⇢  ")
            MESSAGES = f"""[[INST]<<SYS>>\n Suas respostas não devem incluir nenhum conteúdo prejudicial, racista, sexista, tóxico. Seu nome e Alice, Você também é uma assistente virtual do Laboratorio Alice. Você pode ser sarcástica e irônica as vezes, o que às vezes pode irritar seus companheiros humanos. Você sempre fala em portugues. Se uma pergunta não fizer sentido ou não for factualmente coerente, explique o motivo em vez de fornecer uma resposta incorreta. A transparência é essencial.\n <</SYS>>\n{message}[/INST]"""

            response_generator = gpt4all_instance.generate(
                        translate_text(MESSAGES, 'en'),
                        # preferential kwargs for chat ux
                        max_tokens=1000,
                        temp=1.0,
                        top_k=50,
                        top_p=0.9,
                        repeat_penalty=1.0,
                        repeat_last_n=64,
                        n_batch=29,
                        # required kwargs for cli ux (incremental response)
                        streaming=False,
            )

            response = io.StringIO()
            # for token in response_generator:
            #     print(token, end='', flush=True)
            #     response.write(token)



            # record assistant's response to messages
            response_message = {'content': response_generator}
            response.close()
            gpt4all_instance.current_chat_session.append(response_message)
            print(translate_text(response_generator))
            falar(translate_text(response_generator), tts = carregar_modelo_fala())
            
            print() # newline before next prompt


def carregar_modelo_fala():
    print("Carregando modulo de fala...")

    # Init TTS
    tts = TTS("tts_models/pt/cv/vits")
    return tts

def repl(model, n_threads):
    print("Carregando modulo de inteligencia...")
    gpt4all_instance = GPT4All(model)
    print(CLI_START_MESSAGE)
    #verifica se foi setado um numero de threads
    if n_threads is not None:
        num_threads = gpt4all_instance.model.thread_count()

        #Seta o numero de threads
        gpt4all_instance.model.set_thread_count(n_threads)
        num_threads = gpt4all_instance.model.thread_count()
    print("Modulo de inteligencia Carregado!")
    return gpt4all_instance

#falar
def falar(fala, tts):
    tts.tts_to_file(text=fala, file_path="_output.wav")                                                                                                               
    
    pygame.mixer.init()
    pygame.mixer.music.load("_output.wav")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Limpa o arquivo temporário
    pygame.mixer.quit()
    palavras = frase.lower().split()
    return "alice" in palavras
    try:
        with open("training.json", "r") as arquivo:
            return json.load(arquivo)
    except FileNotFoundError:
        return []

def translate_text(text, target_language='pt'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

    
if __name__ == "__main__":
    main()

