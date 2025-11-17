# RobotAgent - LangChain Setup

A LangChain-based agent project for building intelligent agents.

## Setup

1. **Activate the virtual environment:**
   ```bash
   source rvenv/bin/activate
   ```

2. **Install dependencies (if needed):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys:
     - `OPENAI_API_KEY`: Required for OpenAI models
     - `LANGSMITH_API_KEY`: Optional, for observability

4. **Install python-dotenv (for loading .env files):**
   ```bash
   pip install python-dotenv
   ```

## Running Examples

### Simple Chat Example (Recommended to start)
```bash
python simple_example.py
```

### Agent Example (with tools)
```bash
python main.py
```

**Note:** Make sure you have set your `OPENAI_API_KEY` in the `.env` file before running.

## Project Structure

- `main.py` - Basic agent example with a calculator tool
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create from `.env.example`)

## Next Steps

- Explore [LangChain Documentation](https://docs.langchain.com/)
- Check out [LangGraph](https://docs.langchain.com/docs/langgraph) for more advanced agent orchestration
- Set up [LangSmith](https://smith.langchain.com/) for observability and debugging

## Resources

- [LangChain Docs](https://docs.langchain.com/)
- [LangChain Python API Reference](https://api.python.langchain.com/)
- [LangGraph Documentation](https://docs.langchain.com/docs/langgraph)

