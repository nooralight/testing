import telebot
from predicting import Testing
import time
import os
from dotenv import load_dotenv
load_dotenv()

# Loading telegram bot token from the .env file
TOKEN = os.getenv("BOT_TOKEN")

# Initializing the bot with token
bot = telebot.TeleBot(TOKEN)


"""
    In this section, we declared a command which is "/start" . This command means whenever a new user enters to the bot,
    the bot will welcome them with greetings..
    Also if any user send this command message, they will also have this greeting message.
"""
@bot.message_handler(commands=['start'])
def start(message):
    # Getting the message sender's information
    sender = message.from_user
    # Sending message
    bot.send_message(message.chat.id, f"Hi <a href=\"tg://user?id={sender.id}\">{sender.full_name}</a> , thanks for coming here.\nHow can we help you?",parse_mode = "HTML")



"""
    In this section, we declared a command which is "/help". By sending this command message, users can
    have helping message or instructions from the bot.
"""

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, f"Thanks for asking for help to us. Here you can ask your queries to this bot by following this steps: \nType /ask&lt;space&gt;&lt;Your query&gt;\nLike this :\n<code>/ask What is the weather now?</code>\n<code>/ask How are you?</code>",parse_mode = "HTML")

"""
    In this section,
    By this command, user will start chatting with the bot. They can ask dataset related questions, and the bot will answer to 
    that query. 
"""

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    sender = message.from_user
    if sender.full_name:
        sender_text = f"<a href=\"tg://user?id={sender.id}\">{sender.full_name}</a>"
    else:
        sender_text = f"<a href=\"tg://user?id={sender.id}\">{sender.id}</a>"
    # Sending a short temporary message while processing the response. This message will be deleted later
    bot.reply_to(message, f"Processing command by {sender_text}...", parse_mode = "HTML")
    prompt = message.text
    obj = Testing()
    # Getting response from out chatbot model
    result = obj.response(prompt)
    # Sending the bot response to the user
    bot.send_message(message.chat.id, f"{result}",reply_to_message_id= message.message_id)
    # Deleting the temporary message
    bot.delete_message(chat_id= message.chat.id,message_id = message.message_id+1)



## Starting the polling of the telegram bot

bot.polling()