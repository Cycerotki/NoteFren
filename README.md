# NoteFren

Inspired by Open NotebookLM, but wondering if it can be made local.  
NoteFren brings ai-assisted notetaking to everyone who wants to run everything locally. 

## Components  
NoteFren provides the following services to the confused student:  
- Markdown Notes
- Sticky Notes
- Mindmap
- Internet Lookup (via DuckDuckGoSearch)

NoteFren also offers these AI assistants:
- Audio transcription (using Automatic Speech Recognition)
- LLM chat
- LLM tool runner
- LLM Summariser
- OCR (to get text outta images)
- Podcast summary (via Text To Speech)
- PDF parser

## Stack
- Streamlit UI
- FastAPI
- Ollama
- Redis

## I want to try build this myself!
When there is time, tutorials will be written.  
For now, we will list commands in subsections with brief descriptions.  

### Server
1. Create a server folder
2. Go to FastAPI website to make a simple FastAPI server in server folder
3. Run: 
    ```
    cd server
    
    //test environment
    fastapi dev server/main.py

    //production
    fastapi run
    ```
