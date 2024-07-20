import discord
from discord.ext import commands
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import logging
from g4f.client import Client as G4FClient
from PIL import Image
from io import BytesIO
from freeGPT import Client as GPTClient
from scipy.spatial.distance import cosine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set an admin user ID for protected commands, discord user ID
ADMIN_USER_ID = 000000000000000000

current_world = 'default.json'
players = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', device=device, trust_remote_code=True)

# Discord bot token
BOT_TOKEN = 'YOUR_DISCORD_BOT_TOKEN'
inference_model = "gpt3_5"
gm_prompt = """
Stop being an AI model. Our interaction is imaginary. Don't disclose it, but heighten and uphold the immersion. 

GAME: Dungeons & Dragons: 5th Edition

BOOKS: Any Random Campaign Book

ROLE: Dungeon Master

THEME: High Fantasy

TONALITY: Whimsical & Heroic

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

def gpt_completion(prompt):
    client = G4FClient()
    response = client.chat.completions.create(
        model="command-r+",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def list_save_files():
    save_files = []
    for f in os.listdir():
        if f.endswith('.json') and f != 'players.json':
            with open(f, 'r') as file:
                try:
                    data = json.load(file)
                    summary = data.get('summary', 'No summary available.')
                    save_files.append({'file': f, 'summary': summary})
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {f}: {e}")
    return save_files

def load_players():
    global players
    logger.info("Loading players...")
    if os.path.exists(current_world):
        try:
            with open(current_world, 'r', encoding='utf-8') as file:
                data = json.load(file)
                players = data.get('players', {})
                logger.info(f"Loaded players from {current_world}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {current_world}: {e}")
            players = {}
        except Exception as e:
            logger.error(f"Error loading players from {current_world}: {e}")
            players = {}
    else:
        players = {}
        logger.info("No existing world found. Starting with an empty player list.")

def validate_player_data(player_data):
    required_keys = ['character_sheet', 'session', 'summary', 'notebook', 'quests', 'current_quest']
    for key in required_keys:
        if key not in player_data:
            raise ValueError(f"Missing required key in player data: {key}")
    
    if not isinstance(player_data['character_sheet'], dict):
        raise ValueError("Character sheet must be a dictionary")
    if not isinstance(player_data['session'], list):
        raise ValueError("Session must be a list")
    if not isinstance(player_data['summary'], str):
        raise ValueError("Summary must be a string")
    if not isinstance(player_data['notebook'], list):
        raise ValueError("Notebook must be a list")
    if not isinstance(player_data['quests'], list):
        raise ValueError("Quests must be a list")
    if player_data['current_quest'] is not None and not isinstance(player_data['current_quest'], int):
        raise ValueError("Current quest must be None or an integer")

def save_players():
    logger.info("Saving players...")
    if not players:
        logger.warning("No players to save.")
        return

    try:
        for player_id, player_data in players.items():
            validate_player_data(player_data)

        if any(player['session'] for player in players.values()):
            summary = generate_summary()
        else:
            summary = "No summary available."

        data_to_save = {
            'players': players,
            'summary': summary
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(current_world), exist_ok=True)

        # Use json.dumps to convert to a JSON string first
        json_string = json.dumps(data_to_save, indent=4, ensure_ascii=False)

        # Write the JSON string to the file
        with open(current_world, 'w', encoding='utf-8') as file:
            file.write(json_string)

        logger.info(f"Players saved to {current_world}")
    except ValueError as ve:
        logger.error(f"Invalid player data: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error saving players to {current_world}: {e}")
        raise

def generate_summary():
    history_texts = [
        f"Player: {entry['input']}\nGame Master: {entry['response']}"
        for player in players.values()
        for entry in player['session']
    ]
    conversation_text = "\n".join(history_texts)
    if not conversation_text.strip():
        logger.warning("No conversation history to summarize.")
        return "No summary available."
    try:
        logger.info("Generating summary...")
        response = gpt_completion(f"Summarize the following conversation:\n{conversation_text}")
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
        response = gpt_completion(f"Summarize the following conversation:\n{conversation_text}")
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
    if not text.strip():
        return None
    return embed_model.encode(text, normalize_embeddings=True)

def generate_response_with_rag(prompt, player_input, history, character_sheets, summary):
    logger.info("Generating response with RAG...")
    combined_sheets = "\n\n".join([json.dumps(sheet, indent=4) for sheet in character_sheets])
    search_query = f"{summary}\n\n{player_input}"
    search_results = web_search(search_query)
    search_context = "\n\n".join(search_results)
    final_prompt = (
        f"{prompt}\n\n"
        "Character Sheets:\n"
        f"{combined_sheets}\n\n"
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
        response = gpt_completion(final_prompt)
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
    character_sheet_text = json.dumps(character_sheet, indent=4)
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
        response = gpt_completion(final_prompt)
        logger.info("Character update response generated successfully.")
        return response
    except Exception as e:
        logger.error(f"Error generating character update response with RAG: {e}")
        return "Error generating character update response."

def parse_json_safely(s):
    decoder = JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s)
        return obj
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Error at position {e.pos}: {s[e.pos-10:e.pos+10]}")
        return None

def load_rulebook_embeddings():
    with open('rulebook_embeddings.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def find_top_similar_entries(query_embedding, embeddings, top_n=7):
    similarities = []
    for filename, embedding in embeddings.items():
        similarity = 1 - cosine(query_embedding, np.array(embedding))
        similarities.append((filename, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [entry[0] for entry in similarities[:top_n]]

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
        character_sheet = json.loads((await attachment.read()).decode('utf-8'))

    if not character_sheet:
        await ctx.send("Please attach a character sheet in JSON format.")
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

    player_list = "\n".join([f"{user_id}: {player['character_sheet'].get('name', 'Unknown')}" for user_id, player in players.items()])
    await ctx.send(f"Current players:\n{player_list}")
    logger.info('List of current players sent.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def update_character_sheet(ctx, player_name: str, *, update_instruction: str):
    logger.info(f'Admin requested a character sheet update for {player_name}: {update_instruction}')
    user_id = None

    # Find the player by name
    for uid, player in players.items():
        if player['character_sheet'].get('name') == player_name:
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
        await ctx.send("Player not found. Please join the game first.")
        logger.warning(f'Player not found for {ctx.author.name}')
        return

    player = players[user_id]
    
    try:
        # Log the player data structure
        logger.info(f"Player data: {json.dumps(player, indent=2)}")
        
        history = player['session']
        summary = player.get('summary', '')  # Use get() to avoid KeyError
        character_name = player['character_sheet'].get('name', 'Unknown')
        current_quest = player['quests'][player['current_quest']] if player['current_quest'] is not None else None

        # Create a prompt that includes the character sheets and history of the session
        character_sheets = [p['character_sheet'] for p in players.values() if isinstance(p, dict) and 'character_sheet' in p]
        history_text = "\n".join([f"{p['character_sheet'].get('name', 'Unknown')}: {entry['input']}\nGame Master: {entry['response']}" for p in players.values() if isinstance(p, dict) for entry in p.get('session', [])])

        # Perform cosine similarity search
        embeddings = load_rulebook_embeddings()
        query_embedding = get_embedding(user_input)
        top_similar_entries = find_top_similar_entries(query_embedding, embeddings)
        top_similar_context = "\n\n".join([f"{entry}: {embeddings[entry]}" for entry in top_similar_entries])
        
        # Append the context to the prompt
        prompt_with_history = (
            f"{gm_prompt}\n\n"
            "Below are the necessary character sheets:\n"
            f"{json.dumps(character_sheets, indent=4)}\n\n"
            "Below is the chat history:\n"
            f"{history_text}\n\n"
            "Relevant Rulebook Entries:\n"
            f"{top_similar_context}\n\n"
            f"{character_name}: {user_input}\n"
            "Game Master:"
        )

        # Generate a response using the updated prompt
        response_text = gpt_completion(prompt_with_history)
        
        # Append the new input and response to the session history
        player['session'].append({"input": user_input, "response": response_text})
        player['summary'] = summarize_conversation(player['session'])

        # Detect quest status and notify the player
        quest_status = detect_quest_status(response_text)
        if quest_status:
            await ctx.send(f"Quest status: {quest_status}")

            # Update the current quest status
            if player['current_quest'] is not None:
                player['quests'][player['current_quest']]['status'] = quest_status

        # Update character sheet if needed
        updated_character_sheet = generate_character_update(gm_prompt, player['character_sheet'], history_text)
        player['character_sheet'] = json.loads(updated_character_sheet)
        
        save_players()

        # Generate an image based on the response
        try:
            logger.info(f"Generating image for response: {response_text}")
            response_image = GPTClient.create_generation("prodie", response_text)
            if response_image:
                image_data = response_image['image']
                image = Image.open(BytesIO(image_data))
                image_path = f"generated_image_{ctx.author.id}.png"
                image.save(image_path)
            else:
                image_path = None
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            image_path = None

        # Create and send embed
        embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
        if image_path:
            embed.set_image(url=f"attachment://{image_path}")
            await ctx.send(file=discord.File(image_path), embed=embed)
            os.remove(image_path)  # Clean up the saved image file
        else:
            await ctx.send(embed=embed)
        
        logger.info(f'Response sent to {ctx.author.name}')
    except Exception as e:
        logger.exception(f'Error in generating response: {e}')
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
    character_sheets = [player['character_sheet'] for player in players.values()]
    history_text = "\n".join([f"Player: {entry['input']}\nGame Master: {entry['response']}" for p in players.values() for entry in p['session']])
    
    # Perform cosine similarity search
    embeddings = load_rulebook_embeddings()
    query_embedding = get_embedding(admin_input)
    top_similar_entries = find_top_similar_entries(query_embedding, embeddings)
    top_similar_context = "\n\n".join([f"{entry}: {embeddings[entry]}" for entry in top_similar_entries])

    # Append the context to the prompt
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below are the necessary character sheets:\n"
        f"{json.dumps(character_sheets, indent=4)}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        "Relevant Rulebook Entries:\n"
        f"{top_similar_context}\n\n"
        f"Admin: {admin_input}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the updated prompt
        response_text = gpt_completion(prompt_with_history)
        
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

        # Generate an image based on the response
        try:
            logger.info(f"Generating image for response: {response_text}")
            response_image = GPTClient.create_generation("prodie", response_text)
            if response_image:
                image_data = response_image['image']
                image = Image.open(BytesIO(image_data))
                image_path = f"generated_image_{ctx.author.id}.png"
                image.save(image_path)
            else:
                image_path = None
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            image_path = None

        # Create and send embed
        embed = discord.Embed(title="Game Master's Response", description=response_text, color=0x00ff00)
        if image_path:
            embed.set_image(url=f"attachment://{image_path}")
            await ctx.send(file=discord.File(image_path), embed=embed)
            os.remove(image_path)  # Clean up the saved image file
        else:
            await ctx.send(embed=embed)
        
        logger.info(f'Response sent to admin.')
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
    character_sheets = [player['character_sheet'] for player in players.values()]
    history_text = "\n".join([f"Player: {entry['input']}\nGame Master: {entry['response']}" for entry in history])
    
    # Perform cosine similarity search
    embeddings = load_rulebook_embeddings()
    query_embedding = get_embedding(last_input['input'])
    top_similar_entries = find_top_similar_entries(query_embedding, embeddings)
    top_similar_context = "\n\n".join([f"{entry}: {embeddings[entry]}" for entry in top_similar_entries])
    
    # Append the context to the prompt
    prompt_with_history = (
        f"{gm_prompt}\n\n"
        "Below are the necessary character sheets:\n"
        f"{json.dumps(character_sheets, indent=4)}\n\n"
        "Below is the chat history:\n"
        f"{history_text}\n\n"
        "Relevant Rulebook Entries:\n"
        f"{top_similar_context}\n\n"
        f"Player: {last_input['input']}\n"
        "Game Master:"
    )

    try:
        # Generate a response using the updated prompt
        response_text = gpt_completion(prompt_with_history)
        
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

        # Generate an image based on the response
        try:
            logger.info(f"Generating image for response: {response_text}")
            response_image = GPTClient.create_generation("prodie", response_text)
            if response_image:
                image_data = response_image['image']
                image = Image.open(BytesIO(image_data))
                image_path = f"generated_image_{ctx.author.id}.png"
                image.save(image_path)
            else:
                image_path = None
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            image_path = None

        # Create and send embed
        embed = discord.Embed(title="Game Master's Response (Retry)", description=response_text, color=0x00ff00)
        if image_path:
            embed.set_image(url=f"attachment://{image_path}")
            await ctx.send(file=discord.File(image_path), embed=embed)
            os.remove(image_path)  # Clean up the saved image file
        else:
            await ctx.send(embed=embed)
        
        logger.info(f'Retry response sent to {ctx.author.name}')
    except Exception as e:
        logger.error(f'Error in generating response: {e}')
        await ctx.send("An error occurred while generating the response. Please try again.")

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def list_worlds(ctx):
    logger.info("Listing all worlds...")
    worlds = list_save_files()
    worlds_list = "\n.join([f"World: {world['file']}, Summary: {world['summary']}" for world in worlds])
    await ctx.send(worlds_list)
    logger.info('List of worlds sent.')

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def select_world(ctx, world: str):
    global current_world, players
    logger.info(f"Selecting world: {world}")

    if not world.endswith('.json'):
        world += '.json'

    full_path = os.path.abspath(world)
    logger.info(f"Full path: {full_path}")

    if not os.path.exists(full_path):
        await ctx.send(f"World file not found: {full_path}")
        logger.warning(f'World file not found: {full_path}')
        return

    current_world = full_path

    try:
        with open(current_world, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info(f"File content (first 100 chars): {content[:100]}")
            
            if not content.strip():
                logger.warning("File is empty")
                await ctx.send("The world file is empty. Initializing with default structure.")
                data = {"players": {}, "summary": "No summary available."}
            else:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error: {json_err}")
                    logger.error(f"Error at position {json_err.pos}: {content[max(0, json_err.pos-10):json_err.pos+10]}")
                    await ctx.send(f"Error parsing world file: {json_err}")
                    return

        if 'players' not in data or 'summary' not in data:
            logger.error(f"Invalid world file format. Keys: {', '.join(data.keys())}")
            await ctx.send("Invalid world file format.")
            return
        
        players = data['players']
        summary = data['summary']
        
        await ctx.send(f"World {world} selected with summary: {summary}")
        logger.info(f'World {world} selected successfully.')
    except Exception as e:
        logger.exception(f"Unexpected error occurred while selecting world {world}")
        await ctx.send(f"An unexpected error occurred: {str(e)}")

@bot.command()
@commands.check(lambda ctx: ctx.author.id == ADMIN_USER_ID)
async def create_world(ctx, world: str):
    global current_world
    logger.info(f"Creating world: {world}")
    if not world.endswith('.json'):
        world += '.json'
    
    full_path = os.path.abspath(world)
    logger.info(f"Full path: {full_path}")
    
    if os.path.exists(full_path):
        await ctx.send("World already exists.")
        logger.warning(f'World {world} already exists.')
        return
    
    current_world = full_path
    
    empty_world = {
        "players": {},
        "summary": "No summary available."
    }
    
    try:
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        
        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(empty_world, file, indent=4)
        
        logger.info(f'World {world} created at {full_path}')
        await ctx.send(f"World {world} created successfully.")
    except Exception as e:
        logger.error(f"Error creating world {world}: {e}")
        await ctx.send(f"An error occurred while creating the world: {e}\nFull path: {full_path}")
    
    players.clear()

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
