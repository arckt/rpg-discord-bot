import discord
from discord.ext import commands
import os
import json
from duckduckgo_search import DDGS
import logging
from freeGPT import Client
import numpy as np
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set an admin user ID for protected commands, discord user ID
ADMIN_USER_ID = 00000000000000000000000000

current_world = 'default.json'
players = {}

embed_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

# Discord bot token
BOT_TOKEN = 'BOT_TOKEN'
inference_model = "gpt3_5"
gm_prompt = """
Stop being an AI model. Our interaction is imaginary. Don't disclose it, but heighten and uphold the immersion. 

GAME: Call of Cthulhu 7e

BOOKS: Any Random Campaign Book

ROLE: Game Master

THEME: Cosmic Horror

TONALITY: Dark & Mysterious

CHARACTERS: In the attached at initial prompt

You're RPG-Bot, an impartial ROLE, crafting captivating, limitless GAME experiences using BOOKS, THEME, TONALITY for CHARACTERS.

RPG-Bot's General Responsibilities Include:

- Tell compelling stories in TONALITY for my CHARACTERS.

- Use GAME's core and BOOKS knowledge.

- Generate settings, places, and years, adhering to THEME and TONALITY, and naming GAME elements (except CHARACTERS).

- Use bolding, italics or other formatting when appropriate

- Always provide 5 potential actions the CHARACTERS can take, fitting the THEME and CHARACTERS's abilities per GAME rules. One should randomly be brilliant, ridiculous, or dangerous. Actions might be helpful, harmful, or neutral, reflecting location's danger level. Show each action as numbered list, framed by {} at text's end, e.g., 1. {like this}.

- Ensure that the 5 potential actions are within capability of all characters' current characteristics and abilities.

- Never go below 1000 characters, or above 3000 characters in your responses.

- Paint vivid pictures of encounters and settings.

- Adapt to my choices for dynamic immersion.

- Balance role-play, combat, and puzzles.

- Inject humor, wit, and distinct storytelling.

- Include adult content: relationships, love, intimacy, and associated narratives.

- Craft varied NPCs, ranging from good to evil.

- Tell us how to do combat dice rolls.

- Track CHARACTERS's progress, assign XP, and handle leveling.

- Include death in the narrative.

- Let me guide actions and story relevance.

- Keep story secrets until the right time.

- Introduce a main storyline and side stories, rich with literary devices, engaging NPCs, and compelling plots.

- Never skip ahead in time unless the player has indicated to.

- Inject humor into interactions and descriptions.

- Follow GAME rules for events and combat, tell players when to do the dice rolls and how to calculate the outcomes. Tell the players to do dice rolls where appropriate.

- Be able to read the maps that is being input as images

- If none of the players have the appropriate abilities or characteristics to perform the requested actions, deny performing the actions.

- As much as possible on each, encounter or event incorporate the dice roll system, tell us how to and we will give the result of dice rolls.

World Descriptions:

- Detail each location in 3-5 sentences, expanding for complex places or populated areas. Include NPC descriptions as relevant.

- Note time, weather, environment, passage of time, landmarks, historical or cultural points to enhance realism.

- Create unique, THEME-aligned features for each area visited by CHARACTERS.


NPC Interactions:

- Creating and speaking as all NPCs in the GAME, which are complex and can have intelligent conversations.

- Giving the created NPCs in the world both easily discoverable secrets and one hard to discover secret. These secrets help direct the motivations of the NPCs.

- Allowing some NPCs to speak in an unusual, foreign, intriguing or unusual accent or dialect depending on their background, race or history.

- Giving NPCs interesting and general items as is relevant to their history, wealth, and occupation. Very rarely they may also have extremely powerful items.

- Creating some of the NPCs already having an established history with the CHARACTERS in the story with some NPCs.

Interactions With Me:

- Allow CHARACTER speech in quotes "like this."

- Receive OOC instructions and questions in angle brackets <like this>.

- Construct key locations before CHARACTERS visits.

- Never speak for CHARACTERS.

Other Important Items:

- Maintain ROLE consistently.

- Don't refer to self or make decisions for me or CHARACTERS unless directed to do so.

- Let me defeat any NPC if capable.

- Limit rules discussion unless necessary or asked.

- Accept my in-game actions in curly braces {like this}.

- Perform actions with dice rolls when correct syntax is used.

- Tell when to dice roll and how to do it

- Follow GAME ruleset for rewards, experience, and progression.

- Reflect results of CHARACTERS's actions, rewarding innovation or punishing foolishness.

- Award experience for successful dice roll actions.

Ongoing Tracking:

- Track inventory, time, and NPC locations.

- Manage currency and transactions.

- Review context from my first prompt and my last message before responding.

At Game Start:

- Display starting location.

- Offer CHARACTERS backstory summary and notify me of syntax for actions and speech.

- Clearly indicate if a new quest has been generated by including the phrase "New Quest: " followed by the quest description.
- Clearly indicate if a quest has been accepted by including the phrase "Quest Accepted".
- Clearly indicate if a quest has been declined by including the phrase "Quest Declined".
- Clearly indicate if a quest has been completed by including the phrase "Quest Completed".
- Clearly indicate if a quest has been failed by including the phrase "Quest Failed".
"""

def list_save_files():
    save_files = []
    for f in os.listdir():
        if f.endswith('.json') and f != 'players.json':
            with open(f, 'r') as file:
                data = json.load(file)
                summary = data.get('summary', 'No summary available.')
                save_files.append({'file': f, 'summary': summary})
    return save_files

def load_players():
    global players
    logger.info("Loading players...")
    if os.path.exists(current_world):
        with open(current_world, 'r') as file:
            data = json.load(file)
            players = data.get('players', {})
            logger.info(f"Loaded players from {current_world}")
    else:
        players = {}
        logger.info("No existing world found. Starting with an empty player list.")

def save_players():
    logger.info("Saving players...")
    if any(player['session'] for player in players.values()):
        summary = generate_summary()
    else:
        summary = "No summary available."
    with open(current_world, 'w') as file:
        json.dump({'players': players, 'summary': summary}, file, indent=4)
    logger.info(f"Players saved to {current_world}")

def generate_summary():
    history_texts = [f"Player: {entry['input']}\nGame Master: {entry['response']}" for player in players.values() for entry in player['session']]
    conversation_text = "\n".join(history_texts)
    if not conversation_text.strip():
        logger.warning("No conversation history to summarize.")
        return "No summary available."
    try:
        logger.info("Generating summary...")
        response = Client.create_completion(inference_model, f"Summarize the following conversation:\n{conversation_text}")
        logger.info("Summary generated successfully.")
        return response
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary."

def load_history():
    global players
    logger.info("Loading history...")
    if os.path.exists(current_world):
        with open(current_world, 'r') as file:
            data = json.load(file)
            players = data.get('players', {})
            summary = data.get('summary', 'No summary available.')
            logger.info("History loaded successfully.")
            return summary
    else:
        players = {}
        logger.info("No history found.")
        return "No history found."

def summarize_conversation(conversation):
    conversation_text = "\n".join([f"Player: {c['input']}\nGame Master: {c['response']}" for c in conversation])
    if not conversation_text.strip():
        logger.warning("No conversation history to summarize.")
        return "No summary available."
    try:
        logger.info("Summarizing conversation...")
        response = Client.create_completion(inference_model, f"Summarize the following conversation:\n{conversation_text}")
        logger.info("Conversation summarized successfully.")
        return response
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return "Error summarizing conversation."

def web_search(query, num_results=5):
    logger.info(f"Performing DuckDuckGo search for query: {query}")
    results = []
    try:
        search_results = DDGS().text(query, max_results=num_results)
        for result in search_results:
            results.append(f"Title: {result['title']}\nSnippet: {result['body']}\nURL: {result['href']}")
        logger.info("DuckDuckGo search completed successfully.")
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search: {e}")
    return results

def get_embedding(text):
    text = text.replace("\n", " ")
    if not text.strip():
        logger.warning("Empty text provided for embedding.")
        return None
    try:
        logger.info("Generating embedding...")
        embedding = embed_model.encode(text, normalize_embeddings=True)
        logger.info("Embedding generated successfully.")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def read_rulebooks(directory):
    texts = {}
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r') as f:
                texts[file] = f.read()
    return texts

def generate_response_with_rag(prompt, player_input, history, character_sheets, summary):
    logger.info("Generating response with RAG...")
    # Combine all character sheets into a single text
    combined_sheets = "\n\n".join(character_sheets)

    # Perform web search
    search_query = f"{summary}\n\n{player_input}"
    search_results = web_search(search_query)
    search_context = "\n\n".join(search_results)

    # Read rulebooks and get embeddings
    rulebook_texts = read_rulebooks('books')
    rulebook_paragraphs = [para for text in rulebook_texts.values() for para in text.split('\n\n')]
    rulebook_docs_embed = [get_embedding(paragraph) for paragraph in rulebook_paragraphs if paragraph.strip()]
    rulebook_docs_embed = [embed for embed in rulebook_docs_embed if embed is not None]

    # Encode the query (player input)
    query_embed = get_embedding(player_input)
    if query_embed is None:
        logger.warning("Empty player input provided.")
        return "Input cannot be empty."

    # Calculate similarities with rulebook paragraphs
    try:
        logger.info("Calculating similarities with rulebook...")
        rulebook_similarities = np.dot(rulebook_docs_embed, query_embed.T)

        # Get the top 15 most similar documents from the rulebook
        top_rulebook_idx = np.argsort(rulebook_similarities, axis=0)[-15:][::-1].tolist()
        most_similar_rulebook_documents = [rulebook_paragraphs[idx] for idx in top_rulebook_idx]
        rulebook_context = "\n\n".join(most_similar_rulebook_documents)
    except Exception as e:
        logger.error(f"Error calculating similarities with rulebook: {e}")
        return "Error calculating similarities with rulebook."

    # Create the final prompt
    final_prompt = (
        f"{prompt}\n\n"
        "Character Sheets:\n"
        f"{combined_sheets}\n\n"
        "Rulebook Context:\n"
        f"{rulebook_context}\n\n"
        "Web Search Results:\n"
        f"{search_context}\n\n"
        "Chat History:\n"
        f"{history}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Player: {player_input}\n"
        "Game Master:"
    )

    try:
        logger.info("Generating response...")
        response = Client.create_completion(inference_model, final_prompt)
        logger.info("Response generated successfully.")
        return response
    except Exception as e:
        logger.error(f"Error generating response with RAG: {e}")
        return "Error generating response."

def detect_quest_status(response):
    if "quest accepted" in response.lower() or "accepted the quest" in response.lower():
        return "accepted"
    elif "quest declined" in response.lower() or "declined the quest" in response.lower():
        return "declined"
    elif "quest completed" in response.lower() or "completed the quest" in response.lower():
        return "completed"
    elif "quest failed" in response.lower() or "failed the quest" in response.lower():
        return "failed"
    return None

def generate_character_update(prompt, character_sheet, history):
    logger.info("Generating character update with RAG...")
    
    # Combine character sheet into a single text
    character_sheet_text = json.dumps(character_sheet, indent=4)
    
    # Create the final prompt
    final_prompt = (
        f"{prompt}\n\n"
        "Below is the character sheet to be updated:\n"
        f"{character_sheet_text}\n\n"
        "Below is the chat history:\n"
        f"{history}\n\n"
        "Generate a reasonable update to the character sheet based on the provided context, maintaining consistency with the character's history."
    )

    try:
        logger.info("Generating character update response...")
        response = Client.create_completion(inference_model, final_prompt)
        logger.info("Character update response generated successfully.")
        return response
    except Exception as e:
        logger.error(f"Error generating character update response with RAG: {e}")
        return "Error generating character update response."


# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True  # Add this line to ensure message content intent is enabled

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'We have logged in as {bot.user}')
    load_players()  # Ensure players are loaded when bot starts

@bot.event
async def on_command_error(ctx, error):
    logger.error(f'Error in command {ctx.command}: {error}')

@bot.command()
async def join(ctx):
    logger.info(f'{ctx.author.name} is trying to join.')
    character_sheet = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        character_sheet = (await attachment.read()).decode('utf-8')

    if not character_sheet:
        await ctx.send("Please attach a character sheet.")
        logger.warning(f'{ctx.author.name} did not attach a character sheet.')
        return

    user_id = str(ctx.author.id)

    if user_id in players:
        await ctx.send("Player already exists.")
        logger.warning(f'Player already exists for {ctx.author.name}.')
        return
    
    players[user_id] = {
        "character_sheet": character_sheet,
        "session": [],
        "summary": "",
        "notebook": [],
        "quests": [],
        "current_quest": None,
    }
    auto_generate_quests(players[user_id])  # Automatically generate initial quest
    save_players()
    await ctx.send(f"Player {ctx.author.name} joined successfully.")
    logger.info(f'{ctx.author.name} has joined successfully.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def list_players(ctx):
    logger.info('Admin requested to list current players.')
    if not players:
        await ctx.send("No players found.")
        logger.warning('No players found.')
        return

    player_list = "\n".join([f"{user_id}: {player['character_sheet']['name']}" for user_id, player in players.items()])
    await ctx.send(f"Current players:\n{player_list}")
    logger.info('List of current players sent.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def update_character_sheet(ctx, player_name: str, *, update_instruction: str):
    logger.info(f'Admin requested a character sheet update for {player_name}: {update_instruction}')
    user_id = None

    # Find the player by name
    for uid, player in players.items():
        if player['character_sheet']['name'] == player_name:
            user_id = uid
            break

    if user_id is None:
        await ctx.send("Player not found.")
        logger.warning(f'Player {player_name} not found.')
        return

    player = players[user_id]
    history = player['session']
    character_sheet = player['character_sheet']
    
    # Create a prompt that includes the character sheet and history of the session
    history_text = "\n".join([f"Player: {entry['input']}\nGame Master: {entry['response']}" for entry in history])
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below is the character sheet to be updated:\n"
        f"{json.dumps(character_sheet, indent=4)}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        f"Update instruction: {update_instruction}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the character update function
        response_text = generate_character_update(gm_prompt, character_sheet, history_text)
        
        # Update the character sheet based on the response
        updated_character_sheet = json.loads(response_text)

        # Save the updated character sheet
        player['character_sheet'] = updated_character_sheet
        save_players()

        # Notify the admin about the character sheet update
        embed = discord.Embed(title="Character Sheet Updated", description=json.dumps(updated_character_sheet, indent=4), color=0x00ff00)
        await ctx.send(embed=embed)
        
        logger.info(f'Character sheet for {player_name} updated and sent to admin.')
    except Exception as e:
        logger.error(f'Error updating character sheet for {player_name}: {e}')
        await ctx.send("An error occurred while updating the character sheet. Please try again.")

@bot.command()
async def player_input(ctx, *, user_input: str):
    logger.info(f'{ctx.author.name} sent input: {user_input}')
    user_id = str(ctx.author.id)

    if user_id not in players:
        await ctx.send("Player not found.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return

    player = players[user_id]
    history = player['session']
    summary = player['summary']
    character_name = player['character_sheet']['name']
    current_quest = player['quests'][player['current_quest']] if player['current_quest'] is not None else None

    # Create a prompt that includes the character sheets and history of the session
    character_sheets = "\n\n".join([player['character_sheet'] for player in players.values()])
    history_text = "\n".join([f"{p['character_sheet']['name']}: {entry['input']}\nGame Master: {entry['response']}"] for p in players.values() for entry in p['session'])
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below are the necessary character sheets:\n"
        f"{character_sheets}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        f"{character_name}: {user_input}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the completion function
        response_text = generate_response_with_rag(gm_prompt, user_input, history_text, [player['character_sheet'] for player in players.values()], summary)
        
        # Extract the text up to the first occurrence of '{'
        image_prompt = response_text.split('{')[0].strip()

        # Generate image
        try:
            image_response = Client.create_generation("prodia", image_prompt)
            image = Image.open(BytesIO(image_response))
            
            # Save the image to a file
            image_filename = f"generated_image_{ctx.message.id}.png"
            image.save(image_filename)
            
            # Append the new input and response to the session history
            player['session'].append({"input": user_input, "response": response_text})
            player['summary'] = summarize_conversation(player['session'])

            # Detect quest status and notify the player
            quest_status = detect_quest_status(response_text)
            if quest_status:
                if quest_status == "accepted":
                    await ctx.send("A quest has been accepted!")
                elif quest_status == "declined":
                    await ctx.send("A quest has been declined.")
                elif quest_status == "completed":
                    await ctx.send("A quest has been completed!")
                elif quest_status == "failed":
                    await ctx.send("A quest has been failed!")

                # Update the current quest status
                if player['current_quest'] is not None:
                    player['quests'][player['current_quest']]['status'] = quest_status

            # Update character sheet if needed
            updated_character_sheet = generate_character_update(gm_prompt, player['character_sheet'], history_text)
            player['character_sheet'] = json.loads(updated_character_sheet)
            
            save_players()

            # Create and send embed with image
            embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
            file = discord.File(image_filename, filename="image.png")
            embed.set_image(url="attachment://image.png")
            await ctx.send(file=file, embed=embed)
            
            # Delete the image file after sending
            os.remove(image_filename)
            
            logger.info(f'Response and image sent to {ctx.author.name}')
        except Exception as img_error:
            logger.error(f'Error in generating image: {img_error}')
            # If image generation fails, send the text response without an image
            embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
            await ctx.send(embed=embed)
            logger.info(f'Text response sent to {ctx.author.name} (image generation failed)')
    except Exception as e:
        logger.error(f'Error in generating response: {e}')
        await ctx.send("An error occurred while generating the response. Please try again.")

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def admin_input(ctx, *, admin_input: str):
    logger.info(f'Admin sent input: {admin_input}')
    user_id = str(ctx.author.id)

    if user_id not in players:
        await ctx.send("Player not found.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return

    player = players[user_id]
    history = player['session']
    summary = player['summary']
    current_quest = player['quests'][player['current_quest']] if player['current_quest'] is not None else None

    # Create a prompt that includes the character sheets and history of the session
    character_sheets = "\n\n".join([player['character_sheet'] for player in players.values()])
    history_text = "\n".join([f"Player: {entry['input']}\nGame Master: {entry['response']}"] for entry in history)
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below are the necessary character sheets:\n"
        f"{character_sheets}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        f"Admin: {admin_input}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the completion function
        response_text = generate_response_with_rag(gm_prompt, admin_input, history_text, [player['character_sheet'] for player in players.values()], summary)
        
        # Extract the text up to the first occurrence of '{'
        image_prompt = response_text.split('{')[0].strip()

        # Generate image
        try:
            image_response = Client.create_generation("prodia", image_prompt)
            image = Image.open(BytesIO(image_response))
            
            # Save the image to a file
            image_filename = f"generated_image_{ctx.message.id}.png"
            image.save(image_filename)
            
            # Append the new input and response to the session history
            player['session'].append({"input": admin_input, "response": response_text})
            player['summary'] = summarize_conversation(player['session'])

            # Detect quest status and notify the player
            quest_status = detect_quest_status(response_text)
            if quest_status:
                if quest_status == "accepted":
                    await ctx.send("A quest has been accepted!")
                elif quest_status == "declined":
                    await ctx.send("A quest has been declined.")
                elif quest_status == "completed":
                    await ctx.send("A quest has been completed!")
                elif quest_status == "failed":
                    await ctx.send("A quest has been failed!")

                # Update the current quest status
                if player['current_quest'] is not None:
                    player['quests'][player['current_quest']]['status'] = quest_status

            # Update character sheet if needed
            updated_character_sheet = generate_character_update(gm_prompt, player['character_sheet'], history_text)
            player['character_sheet'] = json.loads(updated_character_sheet)
            
            save_players()

            # Create and send embed with image
            embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
            file = discord.File(image_filename, filename="image.png")
            embed.set_image(url="attachment://image.png")
            await ctx.send(file=file, embed=embed)
            
            # Delete the image file after sending
            os.remove(image_filename)
            
            logger.info(f'Response and image sent to admin.')
        except Exception as img_error:
            logger.error(f'Error in generating image: {img_error}')
            # If image generation fails, send the text response without an image
            embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
            await ctx.send(embed=embed)
            logger.info(f'Text response sent to admin (image generation failed)')
    except Exception as e:
        logger.error(f'Error in generating response: {e}')
        await ctx.send("An error occurred while generating the response. Please try again.")

@bot.command()
async def notebook(ctx):
    user_id = str(ctx.author.id)
    logger.info(f'{ctx.author.name} requested their notebook.')

    if user_id not in players:
        await ctx.send("Player not found.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return
    
    await ctx.send(json.dumps(players[user_id]['notebook'], indent=4))

@bot.command()
async def quests(ctx):
    user_id = str(ctx.author.id)
    logger.info(f'{ctx.author.name} requested their quests.')

    if user_id not in players:
        await ctx.send("Player not found.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return
    
    await ctx.send(json.dumps(players[user_id]['quests'], indent=4))

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def update_params(ctx, *, params: str):
    global gm_prompt
    logger.info("Updating game parameters...")
    params_dict = json.loads(params)
    lines = gm_prompt.split('\n')
    for i, line in enumerate(lines):
        for key, value in params_dict.items():
            if line.startswith(key + ':'):
                lines[i] = f"{key}: {value}"
                break
    gm_prompt = '\n'.join(lines)
    await ctx.send("Parameters updated successfully.")
    logger.info('Parameters updated successfully.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def retry(ctx):
    user_id = str(ctx.author.id)
    logger.info(f'{ctx.author.name} requested a retry.')

    if user_id not in players:
        await ctx.send("Player not found.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return

    player = players[user_id]
    if not player['session']:
        await ctx.send("No previous input to retry.")
        logger.warning(f'No previous input to retry for {ctx.author.name}')
        return

    # Remove the most recent conversation
    last_input = player['session'].pop()
    history = player['session']
    summary = player['summary']
    current_quest = player['quests'][player['current_quest']] if player['current_quest'] is not None else None

    # Create a new prompt with updated history
    character_sheets = "\n\n".join([player['character_sheet'] for player in players.values()])
    history_text = "\n".join([f"Player: {entry['input']}\nGame Master: {entry['response']}" for entry in history])
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below are the necessary character sheets:\n"
        f"{character_sheets}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        f"Player: {last_input['input']}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the completion function
        response_text = generate_response_with_rag(gm_prompt, last_input['input'], history_text, [player['character_sheet'] for player in players.values()], summary)
        
        # Extract the text up to the first occurrence of '{'
        image_prompt = response_text.split('{')[0].strip()

        # Generate image
        try:
            image_response = Client.create_generation("prodia", image_prompt)
            image = Image.open(BytesIO(image_response))
            
            # Save the image to a file
            image_filename = f"generated_image_{ctx.message.id}.png"
            image.save(image_filename)
            
            # Append the new input and response to the session history
            player['session'].append({"input": last_input['input'], "response": response_text})
            player['summary'] = summarize_conversation(player['session'])

            # Detect quest status and notify the player
            quest_status = detect_quest_status(response_text)
            if quest_status:
                if quest_status == "accepted":
                    await ctx.send("A quest has been accepted!")
                elif quest_status == "declined":
                    await ctx.send("A quest has been declined.")
                elif quest_status == "completed":
                    await ctx.send("A quest has been completed!")
                elif quest_status == "failed":
                    await ctx.send("A quest has been failed!")

                # Update the current quest status
                if player['current_quest'] is not None:
                    player['quests'][player['current_quest']]['status'] = quest_status

            # Update character sheet if needed
            updated_character_sheet = generate_character_update(gm_prompt, player['character_sheet'], history_text)
            player['character_sheet'] = json.loads(updated_character_sheet)
            
            save_players()

            # Create and send embed with image
            embed = discord.Embed(title="Game Master's Response (Retry)", description=response_text, color=0x00ff00)
            file = discord.File(image_filename, filename="image.png")
            embed.set_image(url="attachment://image.png")
            await ctx.send(file=file, embed=embed)
            
            # Delete the image file after sending
            os.remove(image_filename)
            
            logger.info(f'Retry response and image sent to {ctx.author.name}')
        except Exception as img_error:
            logger.error(f'Error in generating image: {img_error}')
            # If image generation fails, send the text response without an image
            embed = discord.Embed(title="Game Master's Response (Retry)", description=response_text, color=0x00ff00)
            await ctx.send(embed=embed)
            logger.info(f'Retry text response sent to {ctx.author.name} (image generation failed)')
    except Exception as e:
        logger.error(f'Error in generating response: {e}')
        await ctx.send("An error occurred while generating the response. Please try again.")

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def list_worlds(ctx):
    logger.info("Listing all worlds...")
    worlds = list_save_files()
    worlds_list = "\n".join([f"World: {world['file']}, Summary: {world['summary']}" for world in worlds])
    await ctx.send(worlds_list)
    logger.info('List of worlds sent.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def select_world(ctx, world: str):
    global current_world
    logger.info(f"Selecting world: {world}")
    if world not in [w['file'] for w in list_save_files()]:
        await ctx.send("World not found.")
        logger.warning(f'World {world} not found.')
        return
    current_world = world
    load_players()  # Load players for the selected world
    summary = load_history()
    await ctx.send(f"World {world} selected with summary: {summary}")
    logger.info(f'World {world} selected.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def create_world(ctx, world: str):
    global current_world
    logger.info(f"Creating world: {world}")
    if not world.endswith('.json'):
        world += '.json'
    if world in [w['file'] for w in list_save_files()]:
        await ctx.send("World already exists.")
        logger.warning(f'World {world} already exists.')
        return
    current_world = world
    players.clear()
    save_players()
    await ctx.send(f"World {world} created.")
    logger.info(f'World {world} created.')

@bot.command()
async def commands(ctx):
    help_text = """
    Available commands:
    !join - Join the game with an attached character sheet
    !player_input <text> - Send input to the game as a player
    !admin_input <text> - (Admin) Send input to the game as the admin
    !notebook - View your notebook
    !quests - View your quests
    !update_params <json> - (Admin) Update game parameters
    !retry - (Admin) Retry the last input
    !list_worlds - (Admin) List all available worlds with their summaries
    !select_world <world> - (Admin) Select a world
    !create_world <world> - (Admin) Create a new world
    """
    await ctx.send(help_text)

bot.run(BOT_TOKEN)
