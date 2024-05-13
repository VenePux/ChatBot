# Block_1---------------------------------------
# !pip install pytelegrambotapi -q
# !pip install diffusers -q
# !pip install deep-translator -q
#
# !pip install -U g4f --quiet
# !pip install browser-cookie3 --quiet
# !pip install aiohttp_socks --quiet

# Block_2---------------------------------------
import telebot;
bot = telebot.TeleBot('Your token here'); #Telegram bot token from BotFather
from telebot import types
from deep_translator import GoogleTranslator
import g4f
from g4f.Provider import (
    GeekGpt,
    Liaobots,
    Phind,
    Raycast,
    RetryProvider)
from g4f.client import Client
import nest_asyncio
from diffusers import DiffusionPipeline
import torch

# Block_3---------------------------------------
# GPT4
nest_asyncio.apply()

client = Client(
    provider=RetryProvider([
        g4f.Provider.Liaobots,
        g4f.Provider.GeekGpt,
        g4f.Provider.Phind,
        g4f.Provider.Raycast,
        g4f.Provider.BaseProvider,
        g4f.Provider.Bing,
        g4f.Provider.OpenaiChat

    ])
)
chat_history = [{"role": "user", "content": 'Отвечай на русском языке '}]

# Block_4---------------------------------------
def send_request(message):
    global chat_history
    chat_history[0]["content"] += message + " "

    try:
        response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=chat_history
    )
    except Exception as err:
        print("Все провайдеры не отвечают, попробуйте пойзже")
    chat_history[0]["content"] += response + " "
    return response


# Block_5---------------------------------------
#картинка
def send_photo(message):
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16", )
    refiner.to("cuda")


    n_steps = 40
    high_noise_frac = 0.8

    prompt = message

    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent", ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image, ).images[0]
    return image

# Block_6---------------------------------------
# Запуск бота, с зацикливанием, для постоянной работы
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("👋 Привет")
    btn2 = types.KeyboardButton("❓ Задать вопрос")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, 'Привет! Можешь спрашивать меня! Что тебя интересует?', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def func(message):
    if (message.text == "👋 Привет"):
        bot.send_message(message.chat.id, text="Привет! Спасибо, что восьпользовался мною <3")
    elif (message.text == "❓ Задать вопрос"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Как тебя зовут?")
        btn2 = types.KeyboardButton("Что я могу?")
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Задай мне вопрос", reply_markup=markup)

    elif (message.text == "Как тебя зовут?"):
        bot.send_message(message.chat.id, "Меня зовут Гусь. Я самый умный гусь!")

    elif message.text == "Что я могу?":
        bot.send_message(message.chat.id,
                         text="Я могу рассказать о истории создании любого события и нарисовать этот предмет!")

    elif (message.text == "Вернуться в главное меню"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton("👋 Привет")
        button2 = types.KeyboardButton("❓ Задать вопрос")
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text="Вы вернулись в главное меню", reply_markup=markup)
    else:
        inp = GoogleTranslator(source='auto', target='en').translate(message.text)
        inp1 = "Tell me about technology of " + inp
        print(inp1)
        out = GoogleTranslator(source='auto', target='ru').translate(send_request(inp1))
        puu = "photo of the " + inp
        print(puu)
        # Отправка текста
        bot.send_message(message.chat.id, out)
        bot.send_message(message.chat.id, text="Подожди немного и я сгенирирую тебе картинку ♥")
        bot.send_photo(message.chat.id, send_photo(inp))


bot.polling(none_stop=True, interval=0)
