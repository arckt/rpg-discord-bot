# RPG Discord Bot

This repository contains the code for an RPG Discord bot that manages a Call of Cthulhu 7e game. The bot allows players to join the game with their character sheets, interact with the game master (GM) through commands, and manage their game sessions.

## Features

- **Player Management**: Players can join the game by attaching their character sheets.
- **Game Interaction**: Players can send inputs to the game from their character's perspective, and the GM provides responses.
- **Admin Controls**: Admins can update character sheets, retry the last input, and manage game worlds.
- **Quest Management**: Track quests, their statuses, and player progress.
- **Customizable Game Master Prompt**: Modify the GM's prompt to tailor the game experience.
- **Image Generation**: Generate and send images based on game descriptions.
- **Search Integration**: Perform web searches to gather information relevant to the game context.

## Commands

### Player Commands

- `!join`
  - Join the game by attaching a character sheet.
- `!player_input <text>`
  - Send input to the game as your character.
- `!notebook`
  - View your notebook.
- `!quests`
  - View your quests.

### Admin Commands

- `!admin_input <text>`
  - Send input to the game as the admin.
- `!list_players`
  - List all current players.
- `!update_character_sheet <player_name> <update_instruction>`
  - Update a player's character sheet.
- `!update_params <json>`
  - Update game parameters.
- `!retry`
  - Retry the last input.
- `!list_worlds`
  - List all available worlds with their summaries.
- `!select_world <world>`
  - Select a world.
- `!create_world <world>`
  - Create a new world.

### Help Command

- `!commands`
  - List all available commands.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/rpg-discord-bot.git
    cd rpg-discord-bot
    ```

2. Install the required Python modules:

    ```bash
    pip install discord.py duckduckgo-search freeGPT sentence-transformers numpy pillow
    ```
    
3. Run the bot:

    ```bash
    python bot.py
    ```

## Configuration

- **Admin User ID**: Set the `ADMIN_USER_ID` variable in the code to your Discord user ID to restrict certain commands to admins only.
- **Bot Token**: Set `BOT_TOKEN` variable in the code the discord bot token.
- **Rulebooks**: Add a `books` directory with the relevant .txt files for lore.
- **Character Sheets**: Players need to attach their character sheets in a format that suits the group's preferences.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Discord.py](https://github.com/Rapptz/discord.py) for the Discord API wrapper.
- [FreeGPT](https://github.com/free-gpt) for the language model.
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) for search integration.
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embedding generation.
