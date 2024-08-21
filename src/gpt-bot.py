import os
import time
import datetime
import discord
from openai import AsyncOpenAI
from discord.ext import commands
from dotenv import load_dotenv
import tiktoken
from asyncio import Queue
import re
import hashlib
import json
import aiohttp
import aiomysql
import pinecone
import fitz  # PyMuPDF
from docx import Document

load_dotenv()

TOKEN = os.getenv('TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Setup Discord intents
intents = discord.Intents.default()
intents.typing = False
intents.members = True
intents.presences = True
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

bot.message_queue = Queue()

#Initialize OpenAI
openai_client = AsyncOpenAI(api_key = OPENAI_API_KEY)
print(OPENAI_API_KEY)

# Initialize Pinecone
#pinecone.init(api_key=PINECONE_API_KEY, environment=os.getenv('PINECONE_ENVIRONMENT'))

# Create or connect to a Pinecone index
#index_name = os.getenv('PINECONE_NAMESPACE')
#if index_name not in pinecone.list_indexes():
 #   pinecone.create_index(index_name, metric='cosine', dimension=1536)  # OpenAI's embedding vector size
#pinecone_index = pinecone.Index(index_name)

async def get_channel_info(channel):
    description = channel.topic
    members = [f'<@{member.id}> ({member.name}#{member.discriminator})' for member in channel.members if not member.bot]
    return description, members

def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

    script_dir = os.path.dirname(os.path.abspath(__file__))
    roles_file_path = os.path.join(script_dir, "roles", "developer.txt")

    with open(roles_file_path, "r") as file:
        system_message = file.read().strip()

    return system_message

def log_system_message(system_message: str, role_name: str):
    message_hash = hashlib.md5(system_message.encode("utf-8")).hexdigest()
    log_hash_file = "data/logged_system_message_hash"

    if os.environ.get("DEBUG_MODE", "false").lower() == "false":
        return

    if not os.path.exists(log_hash_file):
        with open(log_hash_file, "w") as f:
            f.write(message_hash)
        tokens_count = count_tokens(system_message, "gpt2")
        debug_log(f"Role: {role_name}\nNew System Message:\n{system_message}\nTokens Count: {tokens_count}")
    else:
        with open(log_hash_file, "r") as f:
            logged_hash = f.read().strip()

        if logged_hash != message_hash:
            with open(log_hash_file, "w") as f:
                f.write(message_hash)
            tokens_count = count_tokens(system_message, "gpt2")
            debug_log(f"Role: {role_name}\nNew System Message:\n{system_message}\nTokens Count: {tokens_count}")

async def get_role(channel_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roles_file_path = os.path.join(script_dir, "roles.json")

    with open(roles_file_path, "r") as file:
        roles = json.load(file)

    default_role = os.environ.get("DEFAULT_ROLE") or roles[0]["name"]
    matching_role = next((role for role in roles if role.get("channel_id") == str(channel_id)), None)
    active_role = matching_role or next((role for role in roles if role["name"] == default_role), None)

    return active_role

async def read_system_message(channel):
    role = await get_role(channel.id)
    role_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roles", role["role_file"])

    with open(role_file_path, "r") as file:
        system_message = file.read().strip()

    # Add the "name" property from the role object to the beginning of the system_message
    if role.get("with_name", True):
        system_message = f'Your name is {role["name"]} \n' + system_message

    description, members = await get_channel_info(channel)

    if role.get("add_discord_info", True):
        # Append the additional prompt text, channel description, and members list to the system message
        extra_message = (
            "\n\nYou are a member of your team's Discord channel, respond as a discord user and "
            "answer each request exactly as instructed."
            f'\n\nChannel Members:\n' + '\n'.join(members) +
            f'\n\nChannel Description: {description}'
        )
        system_message += extra_message

    log_system_message(system_message, role["name"])

    return system_message

async def get_gpt_response(system_message, message_history, active_role):
    retries = 3
    delay = 5  # seconds

    while retries > 0:
        try:
            response = await openai_client.chat.completions.create(
                model=active_role.get("model", "gpt-4o"),
                messages=[{"role": "system", "content": system_message}] + message_history,
                max_tokens=active_role.get("output_tokens", 4096),
                n=1,
                stop=None,
                temperature=active_role.get("temperature", 0.8),
            )
            return response
        
        except Exception as e:
            error_msg = f"Error: {e}"
            print(error_msg)
            debug_log(error_msg)
            retries -= 1
            if retries > 0:
                retry_msg = f"Retrying in {delay} seconds... ({retries} retries left)"
                print(retry_msg)
                debug_log(retry_msg)
                time.sleep(delay)
            else:
                return None

async def prune_message_history(message_history, system_message_tokens, input_tokens):
    # Calculate total allowable tokens
    allowable_tokens = input_tokens - system_message_tokens

    # Calculate total tokens from the combined message history
    message_history_tokens = sum(count_tokens(msg["content"], 'gpt2') for msg in message_history)

    # If message history tokens exceed allowable tokens, start truncating
    while message_history_tokens > allowable_tokens:
        # Get the oldest message
        oldest_message = message_history[0]["content"]
        # Calculate the tokens of the oldest message
        oldest_message_tokens = count_tokens(oldest_message, 'gpt2')
        if message_history_tokens - oldest_message_tokens >= allowable_tokens:
            # If removing the oldest message makes the total within the limit, remove it
            message_history.pop(0)
        else:
            # Otherwise, truncate the oldest message as needed
            words = oldest_message.split()
            while message_history_tokens > allowable_tokens and words:
                words.pop()
                new_message = " ".join(words)
                message_history_tokens = sum(count_tokens(msg["content"], 'gpt2') for msg in message_history[:-1]) + count_tokens(new_message, 'gpt2')
            message_history[0]["content"] = new_message
        # Recalculate total tokens after modification
        message_history_tokens = sum(count_tokens(msg["content"], 'gpt2') for msg in message_history)

async def send_message_in_background(channel, text):
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, await send_large_message, channel, text)


def filter_mention(text, bot_id):  # removed async
    bot_mention = f"<@{bot_id}>"
    return text.replace(bot_mention, "")

def debug_log(log_entry: str):
    if os.environ.get("DEBUG_MODE", "false").lower() == "true":
        with open("data/debug.log", "a") as log_file:
            log_file.write(f"{datetime.datetime.now()} | {log_entry}\n")

async def get_relevant_knowledge(prompt, min_score=0.75):  # Default min_score set to 0.75
    # Generate an embedding for the prompt
    embedding = generate_embeddings([prompt])[0]

    # Query the Pinecone index to find the most relevant vectors
    search_results = pinecone_index.query(queries=[embedding], top_k=3)

    if not search_results or 'results' not in search_results:
        print("No search results received from Pinecone.")
        return []

    # Filter matches by the minimum score
    filtered_matches = [match for match in search_results["results"][0]["matches"] if match["score"] >= min_score]

    # Extracting ids (or keys) of the closest vectors
    ids = [match["id"] for match in filtered_matches if 'id' in match]

    # Retrieve the associated text content for these IDs from MySQL
    relevant_texts = await retrieve_texts_by_ids(ids)

    return relevant_texts

async def retrieve_texts_by_ids(ids):
    # Connect to MySQL and retrieve the texts using the provided IDs
    conn = await aiomysql.connect(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT')),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        db=os.getenv('MYSQL_DB')
    )
    results = []
    async with conn.cursor() as cursor:
        for unique_id in ids:
            await cursor.execute("SELECT content FROM knowledge_base WHERE index_id = %s", (unique_id,))
            content = await cursor.fetchone()
            if content:
                results.append(content[0])
    conn.close()
    return results

async def generate_text(prompt, message_history, message_channel):
    system_message = await read_system_message(message_channel)
    system_message_tokens = count_tokens(system_message, 'gpt2')  # calculate system message tokens

    active_role = await get_role(message_channel.id)

    #relevant_knowledge = await get_relevant_knowledge(prompt)

    # Prepend the relevant knowledge to the recent prompt
    #full_prompt = "\n".join(relevant_knowledge + [prompt])
    full_prompt = prompt

    if active_role['message_history']:  # check if message_history is true
        message_history.append({"role": "user", "content": full_prompt})
        await prune_message_history(message_history, system_message_tokens, active_role['input_tokens'])
    else:  # if message_history is false
        message_history = [{"role": "user", "content": full_prompt}]  # set message_history to only contain the recent prompt

    response = await get_gpt_response(system_message, message_history, active_role)

    if response is None:
        return "Sorry, I cannot process your request at the moment. Please try again later."

    response_text = response.choices[0].message.content.strip()
    response_text = filter_mention(response_text, bot.user.id)  # removed await

    if active_role['message_history']:  # check if message_history is true
        message_history.append({"role": "assistant", "content": response_text})

    # Log the prompt and response text using the new debug_log function
    debug_log(f"Prompt: {full_prompt} | Response: {response_text}")

    return response_text

async def generate_and_send_text(prompt, message_history, message_channel):
    async with message_channel.typing():
        response_text = await generate_text(prompt, message_history, message_channel)
        await send_large_message(message_channel, response_text)

async def process_message_queue():
    while True:
        prompt, message_history, message_channel = await bot.message_queue.get()
        await generate_and_send_text(prompt, message_history, message_channel)
        bot.message_queue.task_done()

@bot.event
async def on_ready():
    ready_msg = f'{bot.user.name} has connected to Discord!'
    print(ready_msg)
    debug_log(ready_msg)
    bot.loop.create_task(process_message_queue())

async def send_large_message(channel, text, max_chars=2000):
    code_block_delimiter = "```"
    pattern = f'({code_block_delimiter}.+?{code_block_delimiter})'
    sections = re.split(pattern, text, flags=re.DOTALL)

    for section in sections:
        section_is_code_block = section.startswith(code_block_delimiter) and section.endswith(code_block_delimiter)

        # If it's a code block, strip delimiters
        if section_is_code_block:
            section = section[len(code_block_delimiter):-len(code_block_delimiter)]

        # Split section into smaller chunks and send as individual messages
        while len(section) > max_chars:
            # If it's a code block, we need to take into account the length of the delimiters
            if section_is_code_block:
                split_index = max_chars - len(code_block_delimiter) * 2
            else:
                split_index = section.rfind(' ', 0, max_chars)
                split_index = split_index if split_index != -1 else max_chars

            chunk, section = section[:split_index], section[split_index:].lstrip()

            # If it's a code block, add back the delimiters
            if section_is_code_block:
                chunk = code_block_delimiter + chunk + code_block_delimiter

            await channel.send(chunk)

        # Send the remaining section if it's not empty
        if section.strip():
            # If it's a code block, add back the delimiters
            if section_is_code_block:
                section = code_block_delimiter + section + code_block_delimiter

            await channel.send(section)

# Function to generate embeddings using OpenAI API
def generate_embeddings(text_list):
    response = openai.Embedding.create(
        input=text_list,
        model="text-embedding-ada-002"  # Choose the model that best fits your needs
    )
    return [item['embedding'] for item in response['data']]

# Function to chunk text without cutting off words
def chunk_text(text, chunk_size):
    chunks = []
    while text:
        if len(text) > chunk_size:
            # Find end of last whole word within chunk_size
            cut_off = text.rfind(' ', 0, chunk_size) + 1
            if cut_off <= 0:  # Single word longer than chunk size
                cut_off = text.find(' ') + 1  # Use the first space found
                if cut_off <= 0:  # No spaces found; the single word is the chunk
                    cut_off = len(text)
            chunks.append(text[:cut_off])
            text = text[cut_off:]
        else:
            chunks.append(text)
            break
    return chunks

# Function to generate a unique hash ID for a text chunk
def generate_unique_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Function to chunk text and generate embeddings
async def process_attachment_content(content):
    text_chunks = chunk_text(content, 400)
    embeddings = generate_embeddings(text_chunks)
    return [(generate_unique_id(text), text, embedding) for text, embedding in zip(text_chunks, embeddings)]

# Function to save embeddings and ID-to-content mapping to databases
async def save_to_databases(id_text_embedding_list):
    conn = await aiomysql.connect(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT')),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        db=os.getenv('MYSQL_DB')
    )
    async with conn.cursor() as cursor:
        for unique_id, text, embedding in id_text_embedding_list:
            # Save to MySQL
            await cursor.execute("INSERT INTO knowledge_base (index_id, content) VALUES (%s, %s) ON DUPLICATE KEY UPDATE content = VALUES(content)", (unique_id, text))

            # Check if `embedding` is already a list, if not, convert it to a list
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            # Save to Pinecone
            pinecone_index.upsert(vectors=[(unique_id, embedding_list)])
        await conn.commit()
    conn.close()

# Function to extract text from a PDF byte stream
def extract_text_from_pdf(byte_stream):
    with fitz.open(stream=byte_stream, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Function to extract text from a DOCX byte stream
def extract_text_from_docx(byte_stream):
    doc = Document(byte_stream)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

@bot.command()
async def chat(ctx, *, prompt):
    if not hasattr(bot, 'channel_history'):
        bot.channel_history = {}

    if ctx.channel.id not in bot.channel_history:
        bot.channel_history[ctx.channel.id] = []

    await generate_and_send_text(prompt, bot.channel_history[ctx.channel.id], ctx.channel)

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author.id == bot.user.id:
        return

    if bot.user.mentioned_in(message):
        # Modify the attachment processing part here
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.txt', '.pdf', '.docx')):
                    await message.channel.send("Processing file for knowledge base...")
                    content = None  # Initialize content variable

                    # Download and process the attachment based on its file type
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                file_bytes = await resp.read()

                                if attachment.filename.lower().endswith('.txt'):
                                    content = file_bytes.decode('utf-8')
                                elif attachment.filename.lower().endswith('.pdf'):
                                    content = extract_text_from_pdf(file_bytes)
                                elif attachment.filename.lower().endswith('.docx'):
                                    content = extract_text_from_docx(file_bytes)

                    if content:
                        # Process the content
                        id_text_embedding_list = await process_attachment_content(content)

                        # Save to databases
                        await save_to_databases(id_text_embedding_list)

                        # Post completion message
                        await message.channel.send("File has been processed and added to the knowledge base.")

        if not hasattr(bot, 'message_history'):
            bot.message_history = {}

        active_role = await get_role(message.channel.id)  # Fetch active role based on channel id

        # Fetch recent messages
        recent_messages = []
        async for msg in message.channel.history(limit=50, before=message):
            if msg.author.id != bot.user.id and not msg.author.bot:
                if bot.user.mentioned_in(msg):
                    break
                recent_messages.append(msg)

        # Add the current message (which pinged the bot) to recent_messages
        recent_messages.append(message)

        # Reverse the order of the messages, so the oldest message comes first
        recent_messages.reverse()

        # Combine messages into a chat log
        chat_log = ""
        prev_author = None
        for msg in recent_messages:
            if prev_author is None or prev_author != msg.author:
                if prev_author is not None:
                    chat_log += "\n"
                # Check if add_discord_info is true, then add author mention
                if active_role and 'add_discord_info' in active_role and active_role['add_discord_info']:
                     chat_log += f"<@{msg.author.id}> says:\n"     
                
                prev_author = msg.author

            chat_log += f"{msg.content}\n"

        # Get or create a message history for the current channel
        channel_history = bot.message_history.get(message.channel.id, [])
        bot.message_history[message.channel.id] = channel_history

        # Process the chat log and add it to the message queue
        await bot.message_queue.put((chat_log, channel_history, message.channel))

    await bot.process_commands(message)

@bot.command(name='reset_history')
async def reset_history(ctx):
    # Check if the bot has a message_history attribute
    if hasattr(bot, 'message_history'):
        # Check if the current channel's ID is in the message history
        if ctx.channel.id in bot.message_history:
            # Reset the message history for this channel
            bot.message_history[ctx.channel.id] = []

    await ctx.send('Message history for this channel has been reset.')


@bot.command(name='message_stats')
async def message_stats(ctx):
    # Check if the bot has a message_history attribute
    if hasattr(bot, 'message_history'):
        # Check if the current channel's ID is in the message history
        if ctx.channel.id in bot.message_history:
            message_history = bot.message_history[ctx.channel.id]

            total_tokens = sum(count_tokens(msg["content"], 'gpt2') for msg in message_history)
            total_messages = len(message_history)

            system_message = await read_system_message(ctx.channel)
            system_message_tokens = count_tokens(system_message, 'gpt2')

            report = f"System message tokens: {system_message_tokens}\nTotal message history tokens: {total_tokens}\nTotal messages in history: {total_messages}"
            await ctx.send(report)
        else:
            await ctx.send("No message history for this channel.")
    else:
        await ctx.send("No message history available.")


bot.run(TOKEN)
